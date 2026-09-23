import json

import boto3

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def build_findings_summary(result):
    findings = []
    for a, b, dist in result["duplicate_pairs"]:
        findings.append({"type": "duplicate", "files": [a, b], "distance": dist})
    for path, sharpness in result["blurry_images"]:
        findings.append({"type": "blurry", "file": path, "sharpness": sharpness})
    for path, orientation, swaps_dims in result["risky_orientation_images"]:
        findings.append({
            "type": "risky_exif",
            "file": path,
            "orientation": orientation,
            "swaps_dimensions": swaps_dims,
        })
    return findings


def decide_actions(result):
    findings = build_findings_summary(result)
    if not findings:
        return []

    prompt = (
        "You are reviewing computer vision training data quality findings. "
        "For each finding below, decide \"quarantine\" (remove from the training set) "
        "or \"keep\" (safe to leave in), and give a one-sentence reason.\n\n"
        f"Findings:\n{json.dumps(findings, default=str, indent=2)}\n\n"
        "Respond with ONLY a JSON array like: "
        '[{"file": "...", "action": "quarantine", "reason": "..."}]. '
        "For a duplicate pair, only flag the second file in the pair."
    )

    client = boto3.client("bedrock-runtime")
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    body = json.loads(response["body"].read())
    text = body["content"][0]["text"]
    start = text.find("[")
    end = text.rfind("]")
    return json.loads(text[start:end + 1])


def apply_quarantine(bucket, decisions, key_by_local_path):
    s3 = boto3.client("s3")
    moved = []
    for decision in decisions:
        if decision.get("action") != "quarantine":
            continue
        src_key = key_by_local_path.get(decision["file"])
        if src_key is None:
            continue
        filename = src_key.rsplit("/", 1)[-1]
        dest_key = f"quarantine/{filename}"
        try:
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dest_key)
            moved.append(dest_key)
        except s3.exceptions.ClientError:
            continue
    return moved
