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


def upload_results(bucket, result):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    body = json.dumps(result, default=str, indent=2).encode("utf-8")
    s3 = boto3.client("s3")
    key = f"results/{timestamp}.json"
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key="results/latest.json", Body=body, ContentType="application/json")
    return key


if __name__ == "__main__":
    bucket = os.environ["WINNOW_BUCKET"]
    prefix = os.environ.get("WINNOW_PREFIX", "")

    with tempfile.TemporaryDirectory() as tmp_dir:
        key_by_local_path = download_bucket_prefix(bucket, prefix, tmp_dir)
        result = scan_folder(tmp_dir)

    decisions, quarantined = [], []
    try:
        decisions = decide_actions(result)
        quarantined = apply_quarantine(bucket, decisions, key_by_local_path)
    except Exception as e:
        print("Agent step skipped:", e)

    combined = dict(result)
    combined["agent_decisions"] = decisions
    combined["quarantined"] = quarantined
    result_key = upload_results(bucket, combined)

    print("Duplicate pairs:          ", result["duplicate_pairs"])
    print("Blurry images:            ", result["blurry_images"])
    print("Risky EXIF orientation:   ", result["risky_orientation_images"])
    print("Unreadable images:        ", result["unreadable_images"])
    print("Results written to:       ", f"s3://{bucket}/{result_key}")
    print("Agent decisions:          ", decisions)
    print("Quarantined:              ", quarantined)
