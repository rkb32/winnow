import json
import os

import boto3

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# Bounds the prompt size, and so the cost, of any single scan: 20 identical photos
# would otherwise produce 190 duplicate pairs.
MAX_FINDINGS = 30

# Forcing Claude to answer through a tool call makes Bedrock return schema-shaped JSON,
# instead of free text we'd have to dig a JSON array out of.
DECISION_TOOL = {
    "toolSpec": {
        "name": "record_decisions",
        "description": "Record one keep/quarantine decision per image.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "description": "Image id, e.g. img3"},
                            "action": {"type": "string", "enum": ["quarantine", "keep"]},
                            "reason": {"type": "string", "description": "Under 15 words"},
                        },
                        "required": ["file", "action", "reason"],
                    },
                },
            },
            "required": ["decisions"],
        }},
    }
}


def build_findings_summary(result):
    findings = []
    for train, test, dist in result.get("leaked_pairs", []):
        findings.append({"type": "train_test_leak", "files": [train, test], "distance": dist})
    for train, test, sim, inliers in result.get("similar_leaks", []):
        findings.append({"type": "train_test_leak_edited_copy", "files": [train, test],
                         "similarity": sim, "matching_keypoints": inliers})
    for a, b, dist in result["duplicate_pairs"]:
        findings.append({"type": "duplicate", "files": [a, b], "distance": dist})
    for a, b, sim, inliers in result.get("similar_pairs", []):
        findings.append({"type": "edited_copy", "files": [a, b], "similarity": sim, "matching_keypoints": inliers})
    for path, sharpness in result["blurry_images"]:
        findings.append({"type": "blurry", "file": path, "sharpness": round(sharpness, 1)})
    for path, orientation, swaps_dims in result["risky_orientation_images"]:
        findings.append({
            "type": "risky_exif",
            "file": path,
            "orientation": orientation,
            "swaps_dimensions": swaps_dims,
        })
    return findings


def _display_name(path):
    return os.path.basename(path).split("__")[-1]


def decide_actions(result):
    findings = build_findings_summary(result)[:MAX_FINDINGS]
    if not findings:
        return []

    # Short ids instead of long temp paths keep the prompt and the reply small.
    paths = sorted({p for f in findings for p in f.get("files", [f.get("file")])})
    id_by_path = {p: f"img{i}" for i, p in enumerate(paths)}
    path_by_id = {v: k for k, v in id_by_path.items()}
    for f in findings:
        if "files" in f:
            f["files"] = [id_by_path[p] for p in f["files"]]
        else:
            f["file"] = id_by_path[f["file"]]
    names = {id_by_path[p]: _display_name(p) for p in paths}

    prompt = (
        "You are reviewing computer vision training data quality findings. "
        "Decide \"quarantine\" (remove from the training set) or \"keep\" for each image "
        "that appears in the findings, one decision per image, with a reason under 15 words "
        "that refers to other images by name, never by id. "
        "For a duplicate or edited_copy pair, quarantine only the second image. "
        "For any train_test_leak, quarantine the first (training) image.\n\n"
        f"Image names: {json.dumps(names)}\n"
        f"Findings: {json.dumps(findings)}"
    )

    response = boto3.client("bedrock-runtime").converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        toolConfig={"tools": [DECISION_TOOL], "toolChoice": {"tool": {"name": "record_decisions"}}},
        inferenceConfig={"maxTokens": 1024},
    )
    if response["stopReason"] == "max_tokens":
        raise RuntimeError("Claude's reply was cut off before it finished")

    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            decisions = []
            for d in block["toolUse"]["input"].get("decisions", []):
                path = path_by_id.get(d.get("file"))
                if path is not None:
                    decisions.append({**d, "file": path})
            return decisions
    raise RuntimeError("Claude didn't return any decisions")


def apply_quarantine(bucket, decisions, key_by_local_path, dest_prefix="quarantine/"):
    s3 = boto3.client("s3")
    moved = []
    for decision in decisions:
        if decision.get("action") != "quarantine":
            continue
        src_key = key_by_local_path.get(decision.get("file"))
        if src_key is None:
            continue
        filename = src_key.rsplit("/", 1)[-1]
        dest_key = f"{dest_prefix}{filename}"
        try:
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dest_key)
            moved.append(dest_key)
        except s3.exceptions.ClientError:
            continue
    return moved
