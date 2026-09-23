import base64
import datetime
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
MAX_RUNNING_SCANS = 3
DAILY_SCAN_LIMIT = 15
EXTENSION_BY_TYPE = {"image/jpeg": ".jpg", "image/png": ".png"}
FOLDER_BY_SPLIT = {"train": "images", "test": "test"}
SESSION_RE = re.compile(r"^[0-9a-f]{32}$")

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
    if not cleaned.lower().endswith((".jpg", ".jpeg", ".png")):
        cleaned += EXTENSION_BY_TYPE[content_type]
    return cleaned


def create_upload_urls(body):
    files = body.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        return respond(400, {"error": f"Pick between 1 and {MAX_FILES} photos."})

    session = uuid.uuid4().hex
    uploads = []
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            return respond(400, {"error": "Invalid file list."})
        content_type = f.get("type")
        if content_type not in EXTENSION_BY_TYPE:
            return respond(400, {"error": "Only JPEG and PNG photos are supported."})
        folder = FOLDER_BY_SPLIT.get(f.get("split", "train"))
        if folder is None:
            return respond(400, {"error": "Invalid split."})
        key = f"uploads/{session}/{folder}/{i}/{safe_name(str(f.get('name', '')), content_type)}"
        uploads.append(s3.generate_presigned_post(
            BUCKET,
            key,
            Fields={"Content-Type": content_type},
            Conditions=[{"Content-Type": content_type}, ["content-length-range", 1, MAX_BYTES]],
            ExpiresIn=600,
        ))
    return respond(200, {"session": session, "uploads": uploads})


def start_scan(body):
    session = body.get("session")
    if not isinstance(session, str) or not SESSION_RE.match(session):
        return respond(400, {"error": "Invalid session."})

    listed = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"uploads/{session}/images/", MaxKeys=1)
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
            "environment": [{"name": "WINNOW_SESSION", "value": session}],
        }]},
    )
    if started.get("failures"):
        return respond(503, {"error": "Couldn't start a scan right now, try again in a minute."})
    return respond(202, {"session": session})


ROUTES = {"/upload-urls": create_upload_urls, "/scan": start_scan}


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
