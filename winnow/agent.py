import json
import os

import boto3

from semantic import match_regions
from zoom import clamp_box, crop, thumbnail

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# Bounds the prompt size, and so the cost, of any single scan: 20 identical photos
# would otherwise produce 190 duplicate pairs.
MAX_FINDINGS = 30

# Visual review. In the full Imagenette audit, 12 of the 19 false keypoint matches and 26 of the
# 76 real copies sat at 25-34 inliers (imagenette_exp/review_band.py): in that band the detector's
# own numbers can't tell them apart, so Claude looks at the photos instead of the numbers. Adding
# pairs whose match covers under a quarter of a photo (a shared logo or layout, not the subject)
# catches 15 of the 19.
REVIEW_TYPES = ("edited_copy", "train_test_leak_edited_copy")
REVIEW_MAX_INLIERS = 35
REVIEW_MAX_COVERAGE = 0.25
MAX_REVIEW_PAIRS = 2
# Per pair, not per scan: Claude asks for all its zooms at once, and a shared budget let the
# first pair use every zoom while the second got none. One round, because across 51 audit
# reviews it never asked for a second.
ZOOMS_PER_PAIR = 2
MAX_ZOOM_ROUNDS = 1
# Photos cost tokens, so only scans small enough to have room for them get a review. That keeps
# the worst case per scan near the 30-finding text-only one the daily scan cap was sized for.
REVIEW_MAX_FINDINGS = 8

ZOOM_TOOL = {"toolSpec": {
    "name": "zoom",
    "description": "See one region of an attached photo at full resolution.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "pair": {"type": "string", "description": "Pair id, e.g. pair0"},
            "image": {"type": "string", "enum": ["a", "b"]},
            "box": {"type": "array", "items": {"type": "number"},
                    "description": "[x1, y1, x2, y2] as fractions of the photo's width and height, 0 to 1"},
        },
        "required": ["pair", "image", "box"],
    }},
}}

REVIEW_NOTE = (
    "\n\nThe pairs attached below as photos are borderline: on the detectors' numbers, real matches "
    "and false matches look alike, so check each one yourself. A real match is the same photo, even "
    "cropped, resized, recolored, or mirrored, or another shot of the same moment: the same people, "
    "animals, or objects in the same place, seconds apart, like a burst. A model trained on either "
    "has effectively seen the other, so a real match follows the quarantine rules above. A false "
    "match shares only a template, logo, watermark, layout, or kind of subject: different individual "
    "people, animals, or objects, or the same place on another occasion. If the matched keypoints "
    "cover only a small part of a photo, zoom into the rest to check the subject itself. Give each "
    "attached pair one verdict in pairs: same for a real match, different for a false match, or "
    "unsure if you still can't tell. Leave those pairs' images out of decisions: which one is "
    "removed follows from your verdict."
)


def _decision_tool(review):
    # Forcing Claude to answer through a tool call makes Bedrock return schema-shaped JSON,
    # instead of free text we'd have to dig a JSON array out of.
    properties = {
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
    }
    if review:
        properties["pairs"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair": {"type": "string", "description": "Pair id, e.g. pair0"},
                    "verdict": {"type": "string", "enum": ["same", "different", "unsure"]},
                    "reason": {"type": "string", "description": "Under 15 words, naming what you saw"},
                },
                "required": ["pair", "verdict", "reason"],
            },
        }
    return {"toolSpec": {
        "name": "record_decisions",
        "description": "Record one keep/quarantine decision per image, and one verdict per attached pair.",
        "inputSchema": {"json": {"type": "object", "properties": properties, "required": list(properties)}},
    }}


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


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def select_for_review(findings):
    """The borderline pairs worth showing Claude as photos, the most worth a look first.

    findings: build_findings_summary() entries, with real file paths. Pair entries have "files";
    edited-copy ones also have "similarity" and "matching_keypoints".
    """
    def borderline(f):
        if f["type"] not in REVIEW_TYPES:
            return False
        if f["matching_keypoints"] < REVIEW_MAX_INLIERS:
            return True
        regions = match_regions(*f["files"])
        return regions is not None and min(map(_area, regions)) < REVIEW_MAX_COVERAGE

    # Leaks first, since a wrong call there misstates the test score itself; then the most ambiguous.
    return sorted(filter(borderline, findings),
                  key=lambda f: (f["type"] != "train_test_leak_edited_copy", f["matching_keypoints"]))


def _with_ids(finding, id_by_path):
    finding = dict(finding)
    if "files" in finding:
        finding["files"] = [id_by_path[p] for p in finding["files"]]
    else:
        finding["file"] = id_by_path[finding["file"]]
    return finding


def _image(data):
    return {"image": {"format": "jpeg", "source": {"bytes": data}}}


def _tool_error(use, message):
    return {"toolResult": {"toolUseId": use["toolUseId"], "content": [{"text": message}], "status": "error"}}


def _zoom(use, pairs, budget, zooms):
    request = use["input"]
    pair_id = request.get("pair")
    if not isinstance(pair_id, str) or pair_id not in pairs:
        return _tool_error(use, f"Unknown pair {pair_id!r}.")
    if not budget[pair_id]:
        return _tool_error(use, f"No zooms left for {pair_id}; record your decisions.")
    budget[pair_id] -= 1
    try:
        path = dict(zip("ab", pairs[pair_id]))[request["image"]]
        box = clamp_box(request["box"])
        image, size = crop(path, box)
    # The request is model-written, so a bad one goes back to the model instead of failing the scan.
    except Exception as e:
        return _tool_error(use, f"Invalid zoom request: {e!r}")
    # Only the region is recorded, never pixels: results are public, and the page redraws the
    # crop from the visitor's own copy of the photo.
    zooms.append({"pair": pair_id, "file": path, "box": [round(v, 3) for v in box], "size": list(size)})
    return {"toolResult": {"toolUseId": use["toolUseId"], "content": [_image(image)]}}


def _decisions(use, path_by_id, pairs, targets, zooms):
    answer = use["input"]
    reviewed = {path for pair in pairs.values() for path in pair}
    decisions = {}
    for d in answer.get("decisions", []):
        path = path_by_id.get(d.get("file"))
        if path is not None and path not in reviewed:
            decisions[path] = {**d, "file": path}

    # For borderline pairs Claude only judges whether the photos really match, and code applies the
    # removal rule. Asked for per-photo actions instead, it mixed up which photo to remove. It may
    # dispute a flag but never clear it: a missed leak costs more than a wrongly removed photo, and
    # when it could clear flags it let 15 of 36 real audit leaks through.
    verdicts = {v.get("pair"): v for v in answer.get("pairs", [])}
    for pair_id, pair in pairs.items():
        verdict = verdicts.get(pair_id) or {"verdict": "unsure", "reason": "Claude gave no verdict."}
        pair_zooms = [{k: z[k] for k in ("file", "box", "size")} for z in zooms if z["pair"] == pair_id]
        for path in pair:
            d = decisions.setdefault(path, {"file": path, "action": "keep", "reason": verdict["reason"],
                                            "visual_review": {"verdict": verdict["verdict"], "zooms": []}})
            d["visual_review"]["zooms"] += pair_zooms
            # A file that's the target of two pairs is settled by any one that confirms the match.
            if path == targets[pair_id] and (d["action"] != "quarantine" or verdict["verdict"] == "same"):
                d.update(action="quarantine", reason=verdict["reason"])
                d["visual_review"]["verdict"] = verdict["verdict"]
                if verdict["verdict"] == "same":
                    d.pop("disputed", None)
                else:
                    d["disputed"] = True
    return list(decisions.values())


def decide_actions(result):
    findings = build_findings_summary(result)[:MAX_FINDINGS]
    if not findings:
        return []
    review = select_for_review(findings)[:MAX_REVIEW_PAIRS] if len(findings) <= REVIEW_MAX_FINDINGS else []

    # Short ids instead of long temp paths keep the prompt and the reply small.
    paths = sorted({p for f in findings for p in f.get("files", [f.get("file")])})
    id_by_path = {p: f"img{i}" for i, p in enumerate(paths)}
    path_by_id = {v: k for k, v in id_by_path.items()}
    names = {id_by_path[p]: _display_name(p) for p in paths}

    prompt = (
        "You are reviewing computer vision training data quality findings. "
        "Decide \"quarantine\" (remove from the training set) or \"keep\" for each image "
        "that appears in the findings, one decision per image, with a reason under 15 words "
        "that refers to other images by name, never by id. "
        "For a duplicate or edited_copy pair, quarantine only the second image. "
        "For any train_test_leak, quarantine the first (training) image.\n\n"
        f"Image names: {json.dumps(names)}\n"
        f"Findings: {json.dumps([_with_ids(f, id_by_path) for f in findings])}"
    )
    content = [{"text": prompt + (REVIEW_NOTE if review else "")}]
    pairs, targets = {}, {}
    for k, finding in enumerate(review):
        pair_id, (a, b) = f"pair{k}", finding["files"]
        pairs[pair_id] = (a, b)
        # The file the detector's rule removes: the training copy of a leak, the second copy otherwise.
        targets[pair_id] = a if finding["type"] == "train_test_leak_edited_copy" else b
        label = f"{pair_id}: a = {id_by_path[a]}, b = {id_by_path[b]}."
        regions = match_regions(a, b)
        if regions:
            label += f" Matched keypoints sit in a {regions[0]} and b {regions[1]}."
        content += [{"text": label}, _image(thumbnail(a)), _image(thumbnail(b))]

    client = boto3.client("bedrock-runtime")
    tools = [_decision_tool(bool(review))] + ([ZOOM_TOOL] if review else [])
    messages = [{"role": "user", "content": content}]
    budget, zooms = {pair_id: ZOOMS_PER_PAIR for pair_id in pairs}, []
    for round_number in range(MAX_ZOOM_ROUNDS + 1):
        forced = round_number == MAX_ZOOM_ROUNDS or not any(budget.values())
        choice = {"tool": {"name": "record_decisions"}} if forced else {"any": {}}
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig={"tools": tools, "toolChoice": choice},
            inferenceConfig={"maxTokens": 1024},
        )
        if response["stopReason"] == "max_tokens":
            raise RuntimeError("Claude's reply was cut off before it finished")
        message = response["output"]["message"]
        uses = [block["toolUse"] for block in message["content"] if "toolUse" in block]
        for use in uses:
            if use["name"] == "record_decisions":
                return _decisions(use, path_by_id, pairs, targets, zooms)
        if forced or not uses:
            break
        messages += [message, {"role": "user", "content": [_zoom(use, pairs, budget, zooms) for use in uses]}]
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
