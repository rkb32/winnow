import datetime
import hashlib
import json
import os
import re

import pytest

os.environ.update(
    WINNOW_BUCKET="bucket", WINNOW_CLUSTER="cluster", WINNOW_TASK_DEF="winnow-task",
    WINNOW_SUBNET="subnet-1", WINNOW_SECURITY_GROUP="sg-1", AWS_DEFAULT_REGION="us-east-1",
)
import app  # noqa: E402


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    exceptions = type("Exceptions", (), {"ClientError": ClientError})

    def __init__(self):
        self.photos, self.logged_today, self.already_started = 1, 0, False
        self.logged = []
        self.objects = {}      # kept photos: key -> LastModified
        self.deleted = []

    def generate_presigned_post(self, bucket, key, Fields, Conditions, ExpiresIn):
        return {"url": f"https://{bucket}.s3.amazonaws.com/", "fields": {"key": key, **Fields}, "conditions": Conditions}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://signed.example/{Params['Key']}?expires={ExpiresIn}"

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, Delimiter=None, ContinuationToken=None):
        if Prefix.startswith("saved/"):
            keys = sorted(k for k in self.objects if k.startswith(Prefix))
            if Delimiter:
                folders = sorted({Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter for k in keys})
                return {"KeyCount": len(folders), "CommonPrefixes": [{"Prefix": f} for f in folders][:MaxKeys]}
            return {"KeyCount": len(keys), "Contents": [{"Key": k, "LastModified": self.objects[k]} for k in keys]}
        return {"KeyCount": self.logged_today if Prefix.startswith("scan-log/") else self.photos}

    def put_object(self, Bucket, Key, Body, IfNoneMatch):
        if self.already_started:
            raise ClientError("PreconditionFailed")
        self.logged.append(Key)

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.deleted.append(item["Key"])
            self.objects.pop(item["Key"], None)


class FakeEcs:
    def __init__(self):
        self.running = 0
        self.started = []

    def list_tasks(self, **kwargs):
        return {"taskArns": ["task"] * self.running}

    def run_task(self, **kwargs):
        self.started.append(kwargs)
        return {"tasks": [{}], "failures": []}


@pytest.fixture
def aws(monkeypatch):
    s3, ecs = FakeS3(), FakeEcs()
    monkeypatch.setattr(app, "s3", s3)
    monkeypatch.setattr(app, "ecs", ecs)
    return s3, ecs


def call(path, body, method="POST"):
    raw = body if isinstance(body, str) else json.dumps(body)
    response = app.handler({"rawPath": path, "body": raw, "requestContext": {"http": {"method": method}}}, None)
    return response["statusCode"], json.loads(response["body"])


def photo(name="a.png", type_="image/png", split=None):
    f = {"name": name, "type": type_}
    if split:
        f["split"] = split
    return f


def test_upload_links_are_scoped_to_one_new_session(aws):
    status, body = call("/upload-urls", {"files": [photo(), photo("b.jpg", "image/jpeg", "test")]})
    assert status == 200 and re.fullmatch(r"[0-9a-f]{32}", body["session"])
    train, test = body["uploads"]
    assert train["fields"]["key"] == f"uploads/{body['session']}/images/0/a.png"
    assert test["fields"]["key"] == f"uploads/{body['session']}/test/1/b.jpg"
    assert ["content-length-range", 1, app.MAX_BYTES] in train["conditions"]
    assert {"Content-Type": "image/png"} in train["conditions"]


def test_label_files_get_their_own_folder_and_a_smaller_size_cap(aws):
    labels = [photo("a.txt", "text/plain", "labels"), photo("b.xml", "application/xml", "labels"),
              photo("coco", "application/json", "labels")]
    status, body = call("/upload-urls", {"files": [photo(), *labels]})
    assert status == 200
    keys = [u["fields"]["key"] for u in body["uploads"]]
    assert [k.split("/", 2)[2] for k in keys] == ["images/0/a.png", "labels/1/a.txt", "labels/2/b.xml", "labels/3/coco.json"]
    assert ["content-length-range", 1, app.MAX_LABEL_BYTES] in body["uploads"][1]["conditions"]
    assert ["content-length-range", 1, app.MAX_BYTES] in body["uploads"][0]["conditions"]


def test_label_and_photo_limits_are_counted_separately(aws):
    label = photo("a.txt", "text/plain", "labels")
    assert call("/upload-urls", {"files": [photo()] * app.MAX_FILES + [label] * app.MAX_LABEL_FILES})[0] == 200
    assert call("/upload-urls", {"files": [photo()] + [label] * (app.MAX_LABEL_FILES + 1)})[0] == 400
    assert call("/upload-urls", {"files": [label]})[0] == 400


def test_photo_types_are_not_accepted_as_labels_or_the_reverse(aws):
    assert call("/upload-urls", {"files": [photo(), photo("a.png", "image/png", "labels")]})[0] == 400
    assert call("/upload-urls", {"files": [photo("a.txt", "text/plain")]})[0] == 400


def test_hostile_filenames_cannot_escape_the_session_folder(aws):
    _, body = call("/upload-urls", {"files": [photo("../../<script>.png"), photo("noextension", "image/jpeg")]})
    keys = [u["fields"]["key"] for u in body["uploads"]]
    assert [k.split("/")[-1] for k in keys] == ["script_.png", "noextension.jpg"]
    assert all(k.count("/") == 4 for k in keys)


@pytest.mark.parametrize("body", [
    {"files": [photo("a.gif", "image/gif")]},
    {"files": [photo()] * 21},
    {"files": []},
    {"files": [photo(split="validation")]},
    {"files": ["a.png"]},
    {"files": "a.png"},
])
def test_bad_upload_requests_are_rejected(aws, body):
    assert call("/upload-urls", body)[0] == 400


def test_invalid_json_and_unknown_routes(aws):
    assert call("/upload-urls", "{not json")[0] == 400
    assert call("/upload-urls", [1, 2])[0] == 400
    assert call("/delete-everything", {})[0] == 404
    assert call("/scan", {}, method="GET")[0] == 404


def test_scan_rejects_session_ids_that_could_be_paths(aws):
    assert call("/scan", {"session": "../../results"})[0] == 400


def test_scan_needs_uploaded_photos(aws):
    s3, ecs = aws
    s3.photos = 0
    assert call("/scan", {"session": "0" * 32})[0] == 400
    assert ecs.started == []


def test_daily_cap_stops_new_scans(aws):
    s3, ecs = aws
    s3.logged_today = app.DAILY_SCAN_LIMIT
    assert call("/scan", {"session": "0" * 32})[0] == 429
    assert ecs.started == []


def test_concurrency_cap_stops_new_scans(aws):
    _, ecs = aws
    ecs.running = app.MAX_RUNNING_SCANS
    assert call("/scan", {"session": "0" * 32})[0] == 429
    assert ecs.started == []


def test_repeat_scan_request_for_same_session_starts_nothing(aws):
    s3, ecs = aws
    s3.already_started = True
    assert call("/scan", {"session": "0" * 32})[0] == 202
    assert ecs.started == []


def test_scan_starts_one_task_scoped_to_the_session(aws):
    s3, ecs = aws
    session = "ab" * 16
    assert call("/scan", {"session": session})[0] == 202
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    assert s3.logged == [f"scan-log/{today}/{session}"]
    (task,) = ecs.started
    env = task["overrides"]["containerOverrides"][0]["environment"]
    assert env == [{"name": "WINNOW_SESSION", "value": session}]


# --- kept photos (a guest's past scans) ---

GUEST = "ab" * 32
OTHER_GUEST = "cd" * 32
OWNER = hashlib.sha256(GUEST.encode()).hexdigest()[:32]
SESSION = "1" * 32
NOW = datetime.datetime(2026, 9, 28, tzinfo=datetime.timezone.utc)


def keep(s3, session, *names, owner=OWNER, when=NOW):
    for name in names:
        s3.objects[f"saved/{owner}/{session}/{name}"] = when


def test_kept_uploads_go_under_the_guests_folder_and_never_reveal_the_token(aws):
    status, body = call("/upload-urls", {"files": [photo()], "keep": True, "guest": GUEST})
    assert status == 200
    assert body["uploads"][0]["fields"]["key"] == f"saved/{OWNER}/{body['session']}/images/0/a.png"
    assert GUEST not in json.dumps(body)


def test_uploads_are_not_kept_unless_asked_even_when_a_guest_id_is_sent(aws):
    _, body = call("/upload-urls", {"files": [photo()], "guest": GUEST})
    assert body["uploads"][0]["fields"]["key"].startswith(f"uploads/{body['session']}/")


@pytest.mark.parametrize("guest", [None, "", "short", "zz" * 32, "AB" * 32, 5, "ab" * 31, "ab" * 33])
def test_keeping_photos_needs_a_well_formed_guest_id(aws, guest):
    assert call("/upload-urls", {"files": [photo()], "keep": True, "guest": guest})[0] == 400


def test_a_guest_can_only_keep_so_many_scans(aws):
    s3, _ = aws
    for i in range(app.MAX_KEPT_SCANS):
        keep(s3, f"{i:032x}", "images/0/a.png")
    assert call("/upload-urls", {"files": [photo()], "keep": True, "guest": GUEST})[0] == 400
    assert call("/upload-urls", {"files": [photo()]})[0] == 200


def test_scanning_kept_photos_points_the_task_at_the_guests_folder(aws):
    s3, ecs = aws
    s3.photos = 0
    keep(s3, SESSION, "images/0/a.png")
    assert call("/scan", {"session": SESSION, "guest": GUEST})[0] == 202
    env = ecs.started[0]["overrides"]["containerOverrides"][0]["environment"]
    assert {"name": "WINNOW_BASE", "value": f"saved/{OWNER}/{SESSION}/"} in env


def test_another_guest_cannot_scan_someone_elses_kept_photos(aws):
    s3, ecs = aws
    s3.photos = 0
    keep(s3, SESSION, "images/0/a.png")
    assert call("/scan", {"session": SESSION, "guest": OTHER_GUEST})[0] == 400
    assert call("/scan", {"session": SESSION})[0] == 400
    assert ecs.started == []


def test_photo_links_are_signed_for_photos_only_and_keyed_like_the_dashboard(aws):
    s3, _ = aws
    keep(s3, SESSION, "images/0/a.png", "images/1/b.png", "test/2/c.png", "labels/3/a.txt", "quarantine/b.png")
    status, body = call("/photo-urls", {"session": SESSION, "guest": GUEST})
    assert status == 200 and sorted(body["photos"]) == ["images/0", "images/1", "test/2"]
    assert body["photos"]["images/0"] == f"https://signed.example/saved/{OWNER}/{SESSION}/images/0/a.png?expires=900"


def test_photo_links_are_refused_without_the_right_guest_id(aws):
    s3, _ = aws
    keep(s3, SESSION, "images/0/a.png")
    assert call("/photo-urls", {"session": SESSION, "guest": OTHER_GUEST})[0] == 404
    assert call("/photo-urls", {"session": SESSION})[0] == 400
    assert call("/photo-urls", {"session": "../x", "guest": GUEST})[0] == 400


def test_library_lists_a_guests_scans_newest_first_with_up_to_three_previews(aws):
    s3, _ = aws
    older, newer = "2" * 32, "3" * 32
    keep(s3, older, "images/0/a.png", when=NOW - datetime.timedelta(days=3))
    keep(s3, newer, "images/0/a.png", "images/1/b.png", "images/2/c.png", "images/3/d.png", "test/4/e.png")
    keep(s3, "4" * 32, "images/0/other.png", owner="f" * 32)
    status, body = call("/library", {"guest": GUEST})
    assert status == 200 and [s["session"] for s in body["scans"]] == [newer, older]
    assert body["scans"][0]["photos"] == 5 and len(body["scans"][0]["previews"]) == 3
    assert body["scans"][0]["saved"] == NOW.isoformat()
    assert call("/library", {"guest": OTHER_GUEST})[1] == {"scans": []}
    assert call("/library", {})[0] == 400


def test_forgetting_a_scan_deletes_its_photos_and_nothing_else(aws):
    s3, _ = aws
    keep(s3, SESSION, "images/0/a.png", "test/1/b.png")
    keep(s3, "2" * 32, "images/0/a.png")
    status, body = call("/forget", {"session": SESSION, "guest": GUEST})
    assert status == 200 and body == {"deleted": 2}
    assert list(s3.objects) == [f"saved/{OWNER}/{'2' * 32}/images/0/a.png"]
    assert call("/forget", {"session": SESSION, "guest": OTHER_GUEST})[1] == {"deleted": 0}
