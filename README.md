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

**For borderline pairs, Claude looks before it decides.** When an edited-copy match sits where the detector's own numbers can't separate real copies from false ones (under 35 inliers, or keypoints covering under a quarter of a photo), Claude gets both photos plus two OpenCV tools: a thumbnail, and a full-resolution zoom into any region it picks (`winnow/zoom.py`). It answers one question per pair, same or different, and code applies the removal rule. Claude can **dispute** a flag but never clear one: a disputed photo stays removed unless you click Keep, with Claude's reason and the regions it zoomed into shown on the card. The zoomed regions are stored as coordinates, never pixels, and your browser redraws them from your own copy of the photo.

## Architecture

![Winnow architecture: two entry paths (bulk S3 scan, public website upload) both feed the same ECS Fargate task, which runs the detectors and the Claude agent, writes results to S3, and serves them through CloudFront](winnow/architecture.svg)

Two ways in, one pipeline: an operator with AWS access drops photos in `samples/` and EventBridge triggers a scan (Path A), or anyone uses the public site, which hands out presigned S3 upload links and starts a scoped scan through a Lambda Function URL (Path B). Both land in the same ECS Fargate task — detectors, then the Claude agent, then results in S3 — and both are read back through the same read-only CloudFront-served dashboard.

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

Runs 62 tests inside the exact runtime image (same OpenCV 5 build, same embedding model): every detector, the full `scan_folder` pipeline, Claude's decision handling against a stubbed Bedrock (id mapping, cut-off replies, the findings cap), the visual review loop (per-pair zoom budgets, the forced final decision, bad zoom requests bounced back to the model, the rule that Claude can dispute but not clear a flag), and the upload API's guardrails (hostile filenames, session-id validation, daily and concurrency caps). Production builds skip this stage, so the deployed image carries no test code. To check the tests can actually fail, three deliberate bugs were introduced (a broken hash threshold, a disabled keypoint check, an off-by-one in the daily cap), and the suite caught each one.

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

**Letting Claude look at borderline pairs** (`review_band.py`, `review_eval.py`). Among the audit's keypoint-verified flags, 12 of 19 false matches and 26 of 76 real leaks sat at 25–34 inliers, so the detector's own score can't separate them there. Adding "keypoints cover under a quarter of a photo" catches 3 more false matches (15 of 19). Every audit pair that gate selects (51: 36 real leaks, 15 false matches) was then run through the live agent against real Bedrock and scored against the verified verdicts. It took three versions:

| Version | False matches caught | Real leaks wrongly doubted | Real leaks let through |
|---|---|---|---|
| 1. Claude decides keep/remove per photo | 7/15 | — | **15/36** |
| 2. Claude can only dispute, per photo | 9/15 | 18/36 | 0 |
| 3. Claude gives one verdict per pair; code applies the rule | **9/15** | **2/36** | **0** |

Version 1 was worse than no review: it treated burst shots of the same moment as different photos and let 15 real leaks through. That's why Claude can dispute a flag but never clear one. Version 2's doubts were right only 33% of the time, mostly because Claude mixed up *which* photo the rule removes. Version 3 separates judgment from policy, and 9 of its 11 disputes (82%) were right. It costs $0.005 per reviewed pair, and it was measured on 160px photos, where there's little to zoom into.

**Real leaks found in Imagenette itself, at smaller scale.** Across 500,000 train × test pairs, only 2 unplanted pairs were flagged. Both are genuine cross-split duplicates in Imagenette's official split: the same street performer from one burst of photos, and the same garbage truck from a second angle. That's also why accuracy after cleanup (24.4%) lands just below the "clean" baseline: the baseline was itself slightly inflated by those two.

**Why two stages?** (`semantic_eval.py`, `calibrate_verify.py`) Embeddings alone looked perfect on 4,950 pairs, but at 500,000 pairs different photos of the same subject (two men holding the same kind of fish, two garbage trucks) scored up to 0.906. Adding the SIFT + RANSAC check cut 1,772 look-alike candidates down to those 2 real duplicates, while true copies matched with medians of 84–311 keypoints.

An earlier, smaller check (`run_experiment.py`): 10/10 planted exact leaks caught across 130 photos, 0 false positives.

## Repo layout

- `winnow/` — the actual pipeline (detectors, S3 glue, Bedrock agent, Dockerfile, task definition, dashboard)
- `winnow/api/` — the upload API Lambda and its one-time deploy script
- `spike/` — the original dependency spike proving OpenCV 5 + `img_hash` work on ARM64 before building anything else
- `imagenette_exp/` — the real-data evaluations: leak impact, hash vs. embedding comparison, keypoint-threshold calibration, the full audit, and the visual-review band and evaluation
- `proposal.md` / `Winnow-OpenCV-2026-Proposal.pdf` — the original competition proposal

## Limitations & future work

- **Duplicate comparison is O(n²)**: every image is hashed against every other image. Fine for hundreds of images, too slow for a real dataset of thousands. At scale this would need an approximate-nearest-neighbor index (e.g. FAISS or Annoy) instead of brute-force pairwise comparison.
- **Some thresholds are calibrated, others aren't**: the edited-copy thresholds (0.80 similarity, 25 keypoints) were calibrated on Imagenette, which is still one dataset. `sharpness ≤ 200` is a fixed number, so it depends on resolution: it can flag plain-background shots and miss soft focus in large photos. A per-dataset relative threshold would be better.
- **Flags are for human review, not automatic deletion, at dataset scale**: 20% of the keypoint-stage flags in the full audit were look-alikes sharing a stock template or landmark. The website's previews and Keep/Remove overrides exist for exactly this.
- ~~The keypoint check recomputes features per pair~~ — **fixed**: SIFT features are now cached per photo (`_keypoints_for` in `winnow/semantic.py`), so a photo compared against many others only pays that cost once instead of once per pair. This was the dominant cost in the full audit below.
- **The visual review is only measured at 160px**: all 51 evaluation pairs are Imagenette's small version, and uploads to the site are usually full resolution, where zooming has more to find. Its remaining misses are "same product model, different product photo" (3 chainsaw catalog pairs), the same church on a different day, and two garbage trucks. It only runs on scans with ≤ 8 findings and at most 2 pairs per scan, which keeps a scan's worst-case cost close to the text-only one.
- **Heavily compressed crops can slip through**: a copy that's both cropped and heavily recompressed may fail both the hash and the keypoint check (2 of 75 planted leaks were missed).
- **Photo previews exist only in the tab you scanned from**: they're drawn from your own files in the browser, never from a public copy, so reloading a shared result link shows the report without thumbnails.
- **Bedrock agent has a one-time setup dependency**: AWS requires each account to submit a "use case" form to Anthropic before the model can be invoked; this is a one-time manual step, not something the pipeline can do for itself.
- **Public uploads are capped, not authenticated**: anyone can scan photos without an account, so cost is bounded by hard limits instead: 15 scans per day, 3 running at once, and at most 30 findings sent to Claude per scan (about $4/month even under constant abuse). The tradeoff is that someone who burns the daily cap blocks real visitors until the next day. A production version would put uploads behind real accounts with per-user quotas.
