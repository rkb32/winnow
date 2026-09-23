import datetime
import json
import os
import tempfile

import boto3

from scan import scan_folder
from agent import decide_actions, apply_quarantine


def download_bucket_prefix(bucket, prefix, dest_dir):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    key_by_local_path = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = os.path.join(dest_dir, key.replace("/", "__"))
            s3.download_file(bucket, key, local_path)
            key_by_local_path[local_path] = key
    return key_by_local_path


def upload_results(bucket, result, session=None):
    body = json.dumps(result, default=str, indent=2).encode("utf-8")
    s3 = boto3.client("s3")
    if session:
        key = f"results/sessions/{session}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        return key
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    key = f"results/{timestamp}.json"
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key="results/latest.json", Body=body, ContentType="application/json")
    return key


def run(bucket, prefix, test_prefix, quarantine_prefix):
    with tempfile.TemporaryDirectory() as tmp_dir:
        key_by_local_path = download_bucket_prefix(bucket, prefix, tmp_dir)

        test_dir = None
        if test_prefix:
            test_dir = os.path.join(tmp_dir, "_test_split")
            os.makedirs(test_dir)
            key_by_local_path.update(download_bucket_prefix(bucket, test_prefix, test_dir))

        result = scan_folder(tmp_dir, test_folder_path=test_dir)

    decisions, quarantined = [], []
    try:
        decisions = decide_actions(result)
        quarantined = apply_quarantine(bucket, decisions, key_by_local_path, quarantine_prefix)
    except Exception as e:
        print("Agent step skipped:", e)

    combined = dict(result)
    combined["agent_decisions"] = decisions
    combined["quarantined"] = quarantined
    return combined


if __name__ == "__main__":
    bucket = os.environ["WINNOW_BUCKET"]
    session = os.environ.get("WINNOW_SESSION")
    if session:
        prefix = f"uploads/{session}/images/"
        quarantine_prefix = f"uploads/{session}/quarantine/"
    else:
        prefix = os.environ.get("WINNOW_PREFIX", "")
        quarantine_prefix = "quarantine/"

    try:
        combined = run(bucket, prefix, os.environ.get("WINNOW_TEST_PREFIX"), quarantine_prefix)
    except Exception:
        # Without this, a visitor's page would poll for a result that never arrives.
        if session:
            upload_results(bucket, {"error": "scan failed"}, session)
        raise

    result_key = upload_results(bucket, combined, session)

    print("Duplicate pairs:          ", combined["duplicate_pairs"])
    print("Blurry images:            ", combined["blurry_images"])
    print("Risky EXIF orientation:   ", combined["risky_orientation_images"])
    print("Unreadable images:        ", combined["unreadable_images"])
    if "leaked_pairs" in combined:
        print("Train/test leaked pairs:  ", combined["leaked_pairs"])
    print("Results written to:       ", f"s3://{bucket}/{result_key}")
    print("Agent decisions:          ", combined["agent_decisions"])
    print("Quarantined:              ", combined["quarantined"])
