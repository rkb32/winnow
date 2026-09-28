import base64
import datetime
import hashlib
import json
import os
import re
import uuid

import boto3

BUCKET = os.environ["WINNOW_BUCKET"]
CLUSTER = os.environ["WINNOW_CLUSTER"]
TASK_DEF = os.environ["WINNOW_TASK_DEF"]
SUBNET = os.environ["WINNOW_SUBNET"]
SECURITY_GROUP = os.environ["WINNOW_SECURITY_GROUP"]

MAX_FILES = 20
MAX_BYTES = 5 * 1024 * 1024
# One label file per photo (YOLO, Pascal VOC) or a single COCO file for all of them.
MAX_LABEL_FILES = 20
MAX_LABEL_BYTES = 1024 * 1024
MAX_RUNNING_SCANS = 3
DAILY_SCAN_LIMIT = 15
EXTENSION_BY_TYPE = {"image/jpeg": ".jpg", "image/png": ".png"}
LABEL_EXTENSION_BY_TYPE = {"text/plain": ".txt", "application/xml": ".xml", "application/json": ".json"}
FOLDER_BY_SPLIT = {"train": "images", "test": "test", "labels": "labels"}
SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
# A guest who opts in keeps their photos under saved/<owner>/ (a bucket lifecycle rule ends it after
# 30 days). The owner folder is a hash of a secret token that only the guest's browser holds, so the
# token never appears in a key, a result, or a log, and nobody without it can list or sign a photo.
GUEST_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_KEPT_SCANS = 10
PHOTO_URL_SECONDS = 900
PREVIEW_FOLDERS = ("images", "test")
PREVIEWS_PER_SCAN = 3

s3 = boto3.client("s3")
ecs = boto3.client("ecs")


def respond(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def safe_name(name, content_type):
    cleaned = re.sub(r"[^A-Za-z0-9.-]+", "_", name).strip("._")[:80] or "photo"
    if content_type in EXTENSION_BY_TYPE:
        if not cleaned.lower().endswith((".jpg", ".jpeg", ".png")):
            cleaned += EXTENSION_BY_TYPE[content_type]
    elif not cleaned.lower().endswith(LABEL_EXTENSION_BY_TYPE[content_type]):
        cleaned += LABEL_EXTENSION_BY_TYPE[content_type]
    return cleaned


def owner_of(guest):
    if not isinstance(guest, str) or not GUEST_RE.match(guest):
        return None
    return hashlib.sha256(guest.encode()).hexdigest()[:32]


def list_kept(prefix):
    found, token = [], None
    while True:
        page = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        found += page.get("Contents", [])
        token = page.get("NextContinuationToken")
        if not token:
            return found


def create_upload_urls(body):
    files = body.get("files")
    photo_error = respond(400, {"error": f"Pick between 1 and {MAX_FILES} photos."})
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES + MAX_LABEL_FILES:
        return photo_error
    if not all(isinstance(f, dict) for f in files):
        return respond(400, {"error": "Invalid file list."})
    labels = sum(f.get("split") == "labels" for f in files)
    if not 1 <= len(files) - labels <= MAX_FILES:
        return photo_error
    if labels > MAX_LABEL_FILES:
        return respond(400, {"error": f"Pick at most {MAX_LABEL_FILES} label files."})

    owner = None
    if body.get("keep") is True:
        owner = owner_of(body.get("guest"))
        if owner is None:
            return respond(400, {"error": "Keeping photos needs a valid guest id."})
        kept = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"saved/{owner}/", Delimiter="/", MaxKeys=MAX_KEPT_SCANS)
        if len(kept.get("CommonPrefixes", [])) >= MAX_KEPT_SCANS:
            return respond(400, {"error": f"You're keeping {MAX_KEPT_SCANS} scans already. Delete one first."})

    session = uuid.uuid4().hex
    base = f"saved/{owner}/{session}/" if owner else f"uploads/{session}/"
    uploads = []
    for i, f in enumerate(files):
        is_label = f.get("split") == "labels"
        content_type = f.get("type")
        if content_type not in (LABEL_EXTENSION_BY_TYPE if is_label else EXTENSION_BY_TYPE):
            return respond(400, {"error": "Label files must be .txt, .xml or .json." if is_label
                                 else "Only JPEG and PNG photos are supported."})
        folder = FOLDER_BY_SPLIT.get(f.get("split", "train"))
        if folder is None:
            return respond(400, {"error": "Invalid split."})
        key = f"{base}{folder}/{i}/{safe_name(str(f.get('name', '')), content_type)}"
        uploads.append(s3.generate_presigned_post(
            BUCKET,
            key,
            Fields={"Content-Type": content_type},
            Conditions=[{"Content-Type": content_type},
                        ["content-length-range", 1, MAX_LABEL_BYTES if is_label else MAX_BYTES]],
            ExpiresIn=600,
        ))
    return respond(200, {"session": session, "uploads": uploads})


def start_scan(body):
    session = body.get("session")
    if not isinstance(session, str) or not SESSION_RE.match(session):
        return respond(400, {"error": "Invalid session."})

    base = f"uploads/{session}/"
    owner = owner_of(body.get("guest"))
    if owner and s3.list_objects_v2(Bucket=BUCKET, Prefix=f"saved/{owner}/{session}/images/", MaxKeys=1).get("KeyCount"):
        base = f"saved/{owner}/{session}/"
    listed = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{base}images/", MaxKeys=1)
    if not listed.get("KeyCount"):
        return respond(400, {"error": "No photos were uploaded for this scan."})

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    log_prefix = f"scan-log/{today}/"
    started_today = s3.list_objects_v2(Bucket=BUCKET, Prefix=log_prefix, MaxKeys=DAILY_SCAN_LIMIT)
    if started_today.get("KeyCount", 0) >= DAILY_SCAN_LIMIT:
        return respond(429, {"error": "Today's free scans are used up, try again tomorrow."})

    running = ecs.list_tasks(cluster=CLUSTER, family=TASK_DEF, desiredStatus="RUNNING")
    if len(running["taskArns"]) >= MAX_RUNNING_SCANS:
        return respond(429, {"error": "Busy scanning other uploads, try again in a minute."})

    # One log entry per session per day: it counts toward the cap and stops one upload
    # from being rescanned over and over. IfNoneMatch makes the check-and-write atomic.
    try:
        s3.put_object(Bucket=BUCKET, Key=f"{log_prefix}{session}", Body=b"", IfNoneMatch="*")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "PreconditionFailed":
            return respond(202, {"session": session})
        raise

    started = ecs.run_task(
        cluster=CLUSTER,
        taskDefinition=TASK_DEF,
        launchType="FARGATE",
        count=1,
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": [SUBNET],
            "securityGroups": [SECURITY_GROUP],
            "assignPublicIp": "ENABLED",
        }},
        overrides={"containerOverrides": [{
            "name": "winnow",
            "environment": [{"name": "WINNOW_SESSION", "value": session}]
            + ([{"name": "WINNOW_BASE", "value": base}] if base.startswith("saved/") else []),
        }]},
    )
    if started.get("failures"):
        return respond(503, {"error": "Couldn't start a scan right now, try again in a minute."})
    return respond(202, {"session": session})


def kept_request(body):
    """(owner, session) when the request names a valid guest and session, else None."""
    owner, session = owner_of(body.get("guest")), body.get("session")
    if owner is None or not isinstance(session, str) or not SESSION_RE.match(session):
        return None
    return owner, session


def photo_urls(body):
    request = kept_request(body)
    if request is None:
        return respond(400, {"error": "Invalid request."})
    prefix = "saved/%s/%s/" % request
    urls = {}
    for obj in list_kept(prefix):
        parts = obj["Key"][len(prefix):].split("/")
        if len(parts) == 3 and parts[0] in PREVIEW_FOLDERS:
            urls[f"{parts[0]}/{parts[1]}"] = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": obj["Key"]}, ExpiresIn=PHOTO_URL_SECONDS)
    if not urls:
        return respond(404, {"error": "No kept photos for this scan."})
    return respond(200, {"photos": urls})


def library(body):
    owner = owner_of(body.get("guest"))
    if owner is None:
        return respond(400, {"error": "Invalid request."})
    prefix = f"saved/{owner}/"
    scans = {}
    for obj in sorted(list_kept(prefix), key=lambda o: o["Key"]):
        parts = obj["Key"][len(prefix):].split("/")
        if len(parts) == 4 and parts[1] in PREVIEW_FOLDERS:
            scan = scans.setdefault(parts[0], {"session": parts[0], "photos": 0, "saved": obj["LastModified"], "previews": []})
            scan["photos"] += 1
            scan["saved"] = max(scan["saved"], obj["LastModified"])
            if parts[1] == "images" and len(scan["previews"]) < PREVIEWS_PER_SCAN:
                scan["previews"].append(s3.generate_presigned_url(
                    "get_object", Params={"Bucket": BUCKET, "Key": obj["Key"]}, ExpiresIn=PHOTO_URL_SECONDS))
    newest_first = sorted(scans.values(), key=lambda s: s["saved"], reverse=True)
    return respond(200, {"scans": [{**s, "saved": s["saved"].isoformat()} for s in newest_first]})


def forget(body):
    request = kept_request(body)
    if request is None:
        return respond(400, {"error": "Invalid request."})
    keys = [{"Key": obj["Key"]} for obj in list_kept("saved/%s/%s/" % request)]
    for start in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys[start:start + 1000], "Quiet": True})
    return respond(200, {"deleted": len(keys)})


ROUTES = {"/upload-urls": create_upload_urls, "/scan": start_scan,
          "/photo-urls": photo_urls, "/library": library, "/forget": forget}


def handler(event, context):
    route = ROUTES.get(event.get("rawPath"))
    if route is None or event["requestContext"]["http"]["method"] != "POST":
        return respond(404, {"error": "Not found."})

    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw)
    try:
        body = json.loads(raw)
    except ValueError:
        return respond(400, {"error": "Invalid JSON."})
    if not isinstance(body, dict):
        return respond(400, {"error": "Invalid JSON."})
    return route(body)
