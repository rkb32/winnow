import datetime
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

    def generate_presigned_post(self, bucket, key, Fields, Conditions, ExpiresIn):
        return {"url": f"https://{bucket}.s3.amazonaws.com/", "fields": {"key": key, **Fields}, "conditions": Conditions}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):
        return {"KeyCount": self.logged_today if Prefix.startswith("scan-log/") else self.photos}

    def put_object(self, Bucket, Key, Body, IfNoneMatch):
        if self.already_started:
            raise ClientError("PreconditionFailed")
        self.logged.append(Key)


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
