# Winnow

A data-quality auditor for computer vision training datasets. Built for the [OpenCV AI Competition 2026](https://opencv26.devpost.com/) (OpenCV 5 + AWS).

Point it at a folder of training images and it flags the things that quietly corrupt a model before anyone notices: near-duplicate photos, images duplicated across train/test splits (leakage), blurry images, and EXIF orientation tags that don't match how an image actually displays (which shifts bounding boxes without warning). It never stores or forwards the images themselves, everything runs read-in-place.

**Live dashboard:** https://d1oc3ay9n04ubj.cloudfront.net/ — read-only, shows the results from the last real scan.

## What it catches

| Problem | How | Why it matters |
|---|---|---|
| Duplicate / leaked images | Perceptual hash (`cv2.img_hash.BlockMeanHash`), Hamming-like distance ≤ 5 | A duplicate that leaks from train into test inflates apparent accuracy without the model actually generalizing better |
| Blurry images | Laplacian variance ≤ 200 | Out-of-focus images add noise instead of signal |
| Risky EXIF orientation | Orientation tag ≠ 1 (normal) | If bounding-box coordinates weren't rotated to match, they land in the wrong place after display |

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

## Evaluation

Ran the duplicate/leakage detector against 130 real photos from [Imagenette](https://github.com/fastai/imagenette) (2 classes), with 10 real train images deliberately duplicated into the validation split to simulate the kind of leak that happens by accident on a real team:

**10/10 planted leaks caught, 0 false positives.** See `imagenette_exp/run_experiment.py`.

## Repo layout

- `winnow/` — the actual pipeline (detectors, S3 glue, Bedrock agent, Dockerfile, task definition)
- `spike/` — the original dependency spike proving OpenCV 5 + `img_hash` work on ARM64 before building anything else
- `imagenette_exp/` — the real-data leakage experiment
- `proposal.md` / `Winnow-OpenCV-2026-Proposal.pdf` — the original competition proposal

## Limitations & future work

- **Duplicate comparison is O(n²)**: every image is hashed against every other image. Fine for hundreds of images, too slow for a real dataset of thousands. At scale this would need an approximate-nearest-neighbor index (e.g. FAISS or Annoy) instead of brute-force pairwise comparison.
- **Thresholds are provisional**: `distance ≤ 5` and `sharpness ≤ 200` were set from a mix of synthetic test images and a small real-photo sample, not a broad calibration study. They should be tuned against a larger, more diverse dataset before relying on them for a real decision.
- **No automated test suite yet**: the detectors were validated through manual runs and the Imagenette experiment above, not a checked-in `pytest` suite.
- **Bedrock agent has a one-time setup dependency**: AWS requires each account to submit a "use case" form to Anthropic before the model can be invoked; this is a one-time manual step, not something the pipeline can do for itself.
- **Dashboard is read-only and unauthenticated**: the live link above serves `results/latest.json` to anyone who has it, via CloudFront with no login. Fine for a public demo of non-sensitive sample data; a production version scanning real private datasets would need the dashboard behind real auth instead.
