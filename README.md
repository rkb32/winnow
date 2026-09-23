# Winnow

A data-quality auditor for computer vision training datasets. Built for the [OpenCV AI Competition 2026](https://opencv26.devpost.com/) (OpenCV 5 + AWS).

Point it at a folder of training images and it flags the things that quietly corrupt a model before anyone notices: near-duplicate photos, images duplicated across train/test splits (leakage), blurry images, and EXIF orientation tags that don't match how an image actually displays (which shifts bounding boxes without warning). It never stores or forwards the images themselves, everything runs read-in-place.

**Try it:** https://d1oc3ay9n04ubj.cloudfront.net/ — upload up to 20 photos and get a report in about a minute, no install or account needed.

## What it catches

| Problem | How | Why it matters |
|---|---|---|
| Train/test leaks | Both duplicate checks below, run between the training and test sets | A test photo the model has already seen inflates its score without it generalizing any better |
| Exact and compressed copies | Perceptual hash (`cv2.img_hash.BlockMeanHash`), distance ≤ 5, confirmed by ≥ 8 shared SIFT keypoints or ≥ 0.85 embedding similarity | Duplicates overweight some examples and hide leaks. The confirmation exists because raw hashes collide on low-texture photos (see the full audit below) |
| Edited copies (crops, mirrors, recolors) | Two stages: MobileNetV2 embeddings through OpenCV 5's `cv2.dnn` nominate look-alikes (cosine ≥ 0.80), then SIFT keypoints + RANSAC confirm ≥ 25 points agree on one geometric transform | The perceptual hash catches 0% of cropped or mirrored copies (see Evaluation). Embeddings alone flag different photos of the same kind of thing; the geometric check is what tells "same photo" from "same subject" |
| Blurry images | Laplacian variance ≤ 200 | Out-of-focus images add noise instead of signal |
| Risky EXIF orientation | Orientation tag ≠ 1 (normal) | Labels drawn on the unrotated photo may no longer line up. Winnow flags the risk; it doesn't read label files |

Claude (Haiku 4.5 on Bedrock) then reviews every finding and decides what to quarantine, answering through a forced tool call so its output always matches a fixed schema.

## Architecture

```
S3 bucket (images land in samples/)
   │  EventBridge rule watches samples/* only (never results/*, avoids a trigger loop)
   ▼
ECS Fargate task (ARM64 / Graviton)
   │  scan.py runs the three detectors above
   │  agent.py sends findings to Claude (via Bedrock) → decides quarantine/keep
   │  flagged images get copied to quarantine/ in the same bucket
   ▼
Results written to S3 (results/latest.json + a timestamped copy)
   ▼
dashboard.html reads results/latest.json directly, served publicly via CloudFront (read-only, no credentials on the page)
```

The website adds a second entry point into the same container, for people who don't have AWS access:

```
Visitor picks photos on the site
   │  Lambda (winnow/api/app.py) hands out presigned upload links:
   │  one per file, JPEG/PNG only, ≤ 5 MB, into uploads/<session>/ only
   ▼
Browser uploads straight to S3, then asks the Lambda to start the scan
   │  Lambda checks the cost caps, then starts the Fargate task with WINNOW_SESSION set
   ▼
Same scanner + Claude agent, scoped to that one batch
   ▼
results/sessions/<session>.json, which the page polls for and renders
```

Uploaded photos are never readable publicly (CloudFront can only read `dashboard.html` and `results/`), and S3 lifecycle rules delete uploads after 1 day and reports after 7.

IAM is split into three roles by who needs what: an execution role (lets ECS pull the image and ship logs), a task role (lets the running code read/write S3 and call Bedrock, scoped to one bucket), and an EventBridge invocation role (lets the trigger call `ecs:RunTask`). No role does more than one job.

## Running it

```bash
# Build for ARM64 (Graviton) and push
docker buildx build --platform linux/arm64 -t winnow:arm64 --load winnow/
docker tag winnow:arm64 <your-ecr-repo>:arm64
docker push <your-ecr-repo>:arm64

# Register the task definition (edit task-def.json's account/role ARNs first)
aws ecs register-task-definition --cli-input-json file://winnow/task-def.json

# Run it once against a bucket
aws ecs run-task --cluster winnow-cluster --task-definition winnow-task \
  --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[...],securityGroups=[...],assignPublicIp=ENABLED}"
```

Configuration is via environment variables on the task: `WINNOW_BUCKET`, `WINNOW_PREFIX` (defaults to scanning the whole bucket), and optionally `WINNOW_TEST_PREFIX` to enable cross-split leakage detection against a second folder.

For the bulk S3 path, upload your photos to `samples/`, then upload an empty `samples/_ready` file. EventBridge starts one scan on that marker, so dropping 500 photos means one scan, not 500.

### Tests

```bash
docker build --target test winnow/
```

Runs 43 tests inside the exact runtime image (same OpenCV 5 build, same embedding model): every detector, the full `scan_folder` pipeline, Claude's decision handling against a stubbed Bedrock (id mapping, cut-off replies, the findings cap), and the upload API's guardrails (hostile filenames, session-id validation, daily and concurrency caps). Production builds skip this stage, so the deployed image carries no test code. To check the tests can actually fail, three deliberate bugs were introduced (a broken hash threshold, a disabled keypoint check, an off-by-one in the daily cap), and the suite caught each one.

Dependencies are pinned in `winnow/requirements.txt`, and the embedding model is downloaded at build time with a checksum check (`ADD --checksum`) then trimmed by `winnow/models/make_embedder.py`.

## Evaluation

All on real photos from [Imagenette](https://github.com/fastai/imagenette). Scripts are in `imagenette_exp/` and run inside the same container image the pipeline uses.

**Does leakage actually matter, and does Winnow fix it?** (`leak_impact.py`) 75 of 500 test photos were copied into a 1,000-photo training set as exact, compressed, cropped, and mirrored copies. A nearest-neighbor classifier then scored **30.2% instead of 24.8%, which is 5.4 points of accuracy that don't exist.** Winnow caught **73/75 leaks (97%)**, versus 35/75 for the perceptual hash alone, and removing what it flagged brought accuracy back to 24.4%. The classifier memorizes its training data outright, which exaggerates how real models memorize; the direction of the effect carries over, the exact size won't.

| Leak type | Hash alone | Hash + two-stage check |
|---|---|---|
| Exact copy | 19/19 | 19/19 |
| Resized + JPEG q30 | 16/19 | 18/19 |
| Cropped 75% | 0/19 | 19/19 |
| Mirrored | 0/18 | 17/18 |

**A full audit of Imagenette's official split** (`full_audit.py`, results in `imagenette_exp/full_audit/verified_pairs.csv`). Winnow compared all 9,469 training photos against all 3,925 validation photos, 37.2 million pairs, in 34 minutes on 14 cores, using the same detectors as the live pipeline. It flagged 108 pairs. Every pair was then checked by three independent AI reviewers (Claude agents reading the images, not humans), each with a different lens (neutral, a skeptic hunting for differences, a matcher hunting for shared details). The 6 pairs they disagreed on went to a fourth agent that read the full-resolution originals. Each verdict and its visual evidence is in the CSV, so any pair can be re-checked by a person.

- **76 real leaks, touching 67 validation photos (1.7% of the validation set):** 15 are the same photo, and 61 are the same scene from a burst of shots (same people, same objects, moments apart). A model scored on those photos has effectively seen them in training. Most are garbage trucks (18), springer spaniels (12), and chainsaws (12).
- **The two-stage check was 80% precise (76 of 95).** Its false alarms are mostly stock-photo templates, where the same watermark or brochure layout surrounds a different product, plus the same church photographed on a different day.
- **All 13 raw perceptual-hash matches were false.** Low-texture photos (products on white, silhouettes against the sky) collide at this scale; that includes the only 2 "label conflicts" the audit surfaced, which weren't real. Those 13 had at most 6 shared keypoints and at most 0.80 embedding similarity, while every one of 165 real compressed copies the hash caught shared 9+ keypoints. So hash matches now need ≥ 8 shared keypoints or ≥ 0.85 similarity. That rejects all 13 collisions and keeps every real copy, and a regression test (`test_hash_collision_between_different_photos_is_not_reported`) reproduces the collision with a synthetic pair at hash distance 0.

The audit found a flaw in Winnow itself, and fixing it is the point of running one: at 20 photos the collision is rare, but at dataset scale it's guaranteed.

**Real leaks found in Imagenette itself, at smaller scale.** Across 500,000 train × test pairs, only 2 unplanted pairs were flagged. Both are genuine cross-split duplicates in Imagenette's official split: the same street performer from one burst of photos, and the same garbage truck from a second angle. That's also why accuracy after cleanup (24.4%) lands just below the "clean" baseline: the baseline was itself slightly inflated by those two.

**Why two stages?** (`semantic_eval.py`, `calibrate_verify.py`) Embeddings alone looked perfect on 4,950 pairs, but at 500,000 pairs different photos of the same subject (two men holding the same kind of fish, two garbage trucks) scored up to 0.906. Adding the SIFT + RANSAC check cut 1,772 look-alike candidates down to those 2 real duplicates, while true copies matched with medians of 84–311 keypoints.

An earlier, smaller check (`run_experiment.py`): 10/10 planted exact leaks caught across 130 photos, 0 false positives.

## Repo layout

- `winnow/` — the actual pipeline (detectors, S3 glue, Bedrock agent, Dockerfile, task definition, dashboard)
- `winnow/api/` — the upload API Lambda and its one-time deploy script
- `spike/` — the original dependency spike proving OpenCV 5 + `img_hash` work on ARM64 before building anything else
- `imagenette_exp/` — the real-data evaluations: leak impact, hash vs. embedding comparison, keypoint-threshold calibration
- `proposal.md` / `Winnow-OpenCV-2026-Proposal.pdf` — the original competition proposal

## Limitations & future work

- **Duplicate comparison is O(n²)**: every image is hashed against every other image. Fine for hundreds of images, too slow for a real dataset of thousands. At scale this would need an approximate-nearest-neighbor index (e.g. FAISS or Annoy) instead of brute-force pairwise comparison.
- **Some thresholds are calibrated, others aren't**: the edited-copy thresholds (0.80 similarity, 25 keypoints) were calibrated on Imagenette, which is still one dataset. `sharpness ≤ 200` is a fixed number, so it depends on resolution: it can flag plain-background shots and miss soft focus in large photos. A per-dataset relative threshold would be better.
- **Flags are for human review, not automatic deletion, at dataset scale**: 20% of the keypoint-stage flags in the full audit were look-alikes sharing a stock template or landmark. The website's previews and Keep/Remove overrides exist for exactly this.
- **The keypoint check recomputes features per pair**: fine for 20 photos, but the full audit spent ~30 of its 34 minutes there. Caching each photo's SIFT features once would make it roughly 3× faster.
- **Heavily compressed crops can slip through**: a copy that's both cropped and heavily recompressed may fail both the hash and the keypoint check (2 of 75 planted leaks were missed).
- **Photo previews exist only in the tab you scanned from**: they're drawn from your own files in the browser, never from a public copy, so reloading a shared result link shows the report without thumbnails.
- **Bedrock agent has a one-time setup dependency**: AWS requires each account to submit a "use case" form to Anthropic before the model can be invoked; this is a one-time manual step, not something the pipeline can do for itself.
- **Public uploads are capped, not authenticated**: anyone can scan photos without an account, so cost is bounded by hard limits instead: 15 scans per day, 3 running at once, and at most 30 findings sent to Claude per scan (about $4/month even under constant abuse). The tradeoff is that someone who burns the daily cap blocks real visitors until the next day. A production version would put uploads behind real accounts with per-user quotas.
