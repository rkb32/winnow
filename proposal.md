# OpenCV AI Competition 2026 Proposal

**Winnow: a data quality auditor for computer vision training sets**

---

## 1. Team

**Team name:** Winnow Dev
**Members:** Solo entrant.
**Bio:** Ranjeeth Burujula, Saint Louis, Missouri, United States. Undergraduate
computer science student (BCA, distance program), graduating December 2026.
I work primarily in Python and am currently working through data structures and
algorithms. I picked a problem whose deliverable is measured evidence instead of
a demo, because being accountable for the numbers is how I learn a system
properly. Thank you for the opportunity to build it.
**Hackathon experience:** First entry. Scoped accordingly: Section 7 commits to a
minimum viable set and a feature freeze two weeks before the deadline, so
evidence and documentation are produced with time remaining.

---

## 2. Problem and impact

Vision teams invest in models, GPUs and labeling, and almost nothing in verifying
the data underneath. Published audits found that 3.3% of CIFAR-10 test images and
10% of CIFAR-100 test images have duplicates in their own training sets (Barz &
Denzler, *Do We Train on Test Data? Purging CIFAR of Near-Duplicates*, Journal of
Imaging 2020, arXiv:1902.00423), and an average of at least 3.3% label errors
across ten heavily used benchmark test sets, rising to at least 6% of the
ImageNet validation set (Northcutt et al., *Pervasive Label Errors in Test Sets
Destabilize Machine Learning Benchmarks*, NeurIPS 2021, arXiv:2103.14749).
Internal commercial datasets are worse.

Leakage is the expensive defect because its symptom is good news. The model is
graded on memorized material, the number is fiction, and the failure surfaces in
production. No model work recovers from an evaluation set that lies.

**Winnow audits a dataset in place and returns the specific files causing damage,
ranked by severity.** Severity is a stated policy, not an inferred quantity:
test-split leakage outranks annotation errors, which outrank train duplicates,
which outrank quality defects. That ordering is calibrated once by the
dose-response study in §6.

**Relationship to existing data-quality tools.** Confident-learning tools such as
Cleanlab surface likely label errors and outliers from a trained model's
predictions or embeddings, which means a model has to exist and run over the
data first. Winnow operates upstream of that: every defect in §3 is computed
directly from pixels and file metadata, with no model, no embeddings and no
training run required. It is a pre-flight check a team can run before a single
GPU-hour is spent, and it targets a different failure class entirely, image
integrity, duplication and provenance, rather than label-confidence anomalies.

**Privacy property.** Winnow never stores customer pixels. Images are read in
place through a scoped, revocable, read-only cross-account role. Each image
yields a small metrics row and the pixels are discarded. Dashboard thumbnails
render from presigned URLs against the customer's own bucket, so bytes travel
from the customer to the customer's browser and never through Winnow. This makes
it deployable by teams who cannot export imagery, a constraint embedding-based
tools do not address since they still require the images to reach a model.

---

## 3. OpenCV 5 analysis

**Per-image (parallel):**
- **Decode integrity.** `imdecode` on the byte stream; truncated files flagged, not fatal.
- **EXIF orientation conflict.** Decoded twice, once with `IMREAD_IGNORE_ORIENTATION`, because OpenCV applies the tag automatically and would otherwise mask the defect. The raw tag is compared against decoded dimensions and against the annotation coordinate space. A mismatch invalidates every box on that image, and stays invisible to spot-checking because labeling tools honor the tag while most training loaders do not.
- **Blur.** Laplacian variance, cross-checked against Sobel gradient energy, computed tile-wise so partial defocus is not averaged away.
- **Exposure.** Histogram clipping fraction, entropy, dynamic range.
- **Screen photographs.** DFT magnitude spectrum with radial and angular energy profiling to catch display moiré.
- **Annotation plausibility.** Geometry first: out-of-bounds, degenerate area, implausible aspect ratio. Then content, framed as class-conditional outlier detection instead of emptiness. Edge density, Laplacian variance and colour moments inside each box are compared against that class's distribution. Asking whether a box is empty misfires on sky, snow, water and small distant objects. Asking whether a box is unlike others of its class yields a calibrated review queue.

**Pairwise (duplicates and leakage):**
- **Fingerprints.** `img_hash`: 256-bit block-mean as primary, with pHash and colour-moment hashes as complementary signals.
- **Candidate generation.** O(n²) is five billion comparisons at 100k images and does not finish. The hash is split into bands, each indexed separately, and two images become candidates only if they collide in at least one band. Band count derives from the target Hamming radius rather than guesswork: *b* bands guarantee recall within radius *b−1*, which is why the fingerprint is 256 bits, enough bands to cover a realistic near-duplicate radius. Runtime is approximately linear.
- **Cascaded verification.** Full-hash Hamming distance, then dimension and aspect agreement, then ORB with RANSAC homography on the survivors only. ORB costs tens of milliseconds per pair, so an uncascaded candidate net turns into hours. Descriptors are computed once during the scan pass and stored compactly; they are derived features, not pixels, so the privacy property holds. Verification separates true duplicates, including crops, rescales and watermarked re-uploads, from hash collisions.
- **Leakage.** Fingerprints carry split membership, so leakage is a duplicate cluster spanning train and test. Same computation, separate report, highest consequence.

**Scope control.** Only annotation checks need labels. Label ingestion is
optional and isolated behind two adapters, COCO JSON and YOLO text, normalizing
into one internal schema. Other formats are out of scope. If ingestion slips, the
highest-value findings still ship.

---

## 4. AWS architecture

- **Onboarding.** The customer creates a read-only cross-account role scoped to one bucket prefix, trusting Winnow's account ID and gated by an external ID, assumed via STS. They revoke it unilaterally and see every access in their own CloudTrail. Demonstrated across two separate AWS accounts.
- **Control plane.** API Gateway with Lambda handlers; job state in DynamoDB.
- **Fan-out.** A lister shards keys into batches of roughly 2,000 images onto SQS. Batches are large by design: Fargate does not cache container images between tasks, so the OpenCV pull has to be amortized across substantial work.
- **Scan workers (COOL path).** The OpenCV 5 container runs on ECS Fargate ARM64 / Graviton, scaling on queue depth. The work is CPU-bound, memory-light and embarrassingly parallel, which is the Graviton profile. Headless multi-stage build keeps the image small.
- **Fault isolation.** Two retries, then a dead-letter queue, with those images reported as undecodable. One corrupt file degrades a batch, never a scan.
- **Results.** Per-image metric rows, no pixels, written to S3 as Parquet partitioned by scan.
- **Reduce stage, single-task by design.** 100k images produce about 3 MB of fingerprints, and 10M still fit in one container's memory. A distributed index was considered and rejected on arithmetic: it adds cost, failure modes and hot-partition risk for no benefit. The scale-out path is documented for when it is actually needed.
- **Agent (Agentic Vision path).** Bedrock. *Perceive:* aggregate defect statistics and cluster structure, metrics only, never imagery. *Decide:* which clusters materially threaten accuracy, remediation order, auto-quarantine versus human review, and whether a shard warrants a deeper second pass. *Act:* ranked plan, quarantine manifest of object keys, targeted re-scan trigger. A constrained tool interface (`query_metrics`, `sample_cluster`, `request_deep_scan`, `emit_manifest`) keeps actions enumerable and logged, and every recommendation cites its evidence rows. The model is pinned, temperature is 0, and prompts and responses are logged to S3 so a published result replays. A deterministic rule-based ranking sits underneath as both fallback and measured baseline.
- **Presentation.** React dashboard on S3 behind CloudFront, public URL, preloaded scan. No account, no setup.
- **Reproducibility.** AWS CDK, locked Python dependencies, container pinned by digest, seeded corruption, so every published number regenerates.

---

## 5. Architecture

```
Customer AWS account                    Winnow AWS account
┌──────────────────────┐
│  S3: image dataset   │
│  IAM role (read-only,│◄──── STS AssumeRole (external ID, expiring creds)
│  external ID)        │                    │
└──────────────────────┘                    │
    ▲        ▲                              │
    │        │ read in place,       ┌───────┴────────┐
    │        │ pixels discarded     │  API Gateway   │
    │        │                      │  + Lambda      │──► DynamoDB (job state)
    │        │                      └───────┬────────┘
    │        │                              │
    │        │                      ┌───────▼────────┐
    │        │                      │  Lister →  SQS │──► DLQ (poison batches)
    │        │                      └───────┬────────┘
    │        │                              │
    │        │      ┌───────────────────────▼─────────────────┐
    │        └──────┤  ECS Fargate: ARM64 / Graviton          │
    │               │  OpenCV 5 detectors + fingerprinting    │  ◄── COOL path
    │               └───────────────────────┬─────────────────┘
    │                                       │ metric rows (no pixels)
    │                            ┌──────────▼──────────┐
    │                            │  S3 (Parquet)       │
    │                            └──────────┬──────────┘
    │                                       │
    │                     ┌─────────────────▼──────────────────┐
    │                     │ Single reduce task:                │
    │                     │ band index → candidates →          │
    │                     │ cascaded ORB/RANSAC verification   │
    │                     │ → duplicate & leakage clusters     │
    │                     └─────────────────┬──────────────────┘
    │                                       │
    │                     ┌─────────────────▼──────────────────┐
    │                     │ Bedrock agent (metrics only)       │  ◄── Agentic
    │                     │ perceive → decide → act            │      path
    │                     │ ranked plan, quarantine manifest,  │
    │                     │ re-scan trigger                    │
    │                     └─────────────────┬──────────────────┘
    │                                       │
    │                     ┌─────────────────▼──────────────────┐
    │                     │ CloudFront + S3 dashboard          │
    └─────────────────────┤ (thumbnails via presigned URLs     │
      browser fetches     │  straight from customer bucket)    │
      images directly     └────────────────────────────────────┘
```

---

## 6. Users and evaluation

**Primary user and demo vertical: industrial inspection.** Manufacturing QA teams
train defect-detection models (scratches, cracks, dents, misalignment) on photos
of parts from the production line. These datasets are collected in bursts from a
fixed camera rig, which makes them unusually prone to near-duplicates, and
defect classes are rare enough that a single leaked or mislabeled image can
swing a reported accuracy meaningfully. The demo and evaluation datasets in §6
are drawn from public surface and part defect benchmarks (for example MVTec AD
and the NEU surface defect dataset) so the corruption harness operates on
realistic industrial imagery rather than generic photographs.

**Secondary users.** ML platform and data engineers training vision models on
other proprietary imagery, including insurtech, retail catalog, agritech and
autonomous systems, plus labs and dataset publishers certifying a release. The
detectors in §3 are domain-agnostic; industrial inspection is the vertical the
proposal commits to for the working demo and dataset choice, not a limitation of
what the system can audit. The motivation across all of these is the same and
narrow: knowing whether the accuracy number about to be presented is real.

**Detector correctness.** A seeded harness injects a known manifest into clean
public data, using Pascal VOC and a COCO subset for annotations and Imagenette
for training runs. Injected defects are duplicates, cross-split copies, graduated
blur, conflicting EXIF orientation (added, not stripped, since public datasets
arrive normalized), displaced annotations and simulated screen photographs.
Ground truth is authored, so per-defect precision, recall and F1, plus duplicate
cluster purity, are directly computable.

**Index correctness.** The band index is an optimization and has to be shown not
to lose results. On a 10,000-image subset the exhaustive O(n²) comparison is run
once and the fast path's recall is measured against it.

**Two accuracy claims, tested separately.** They differ in magnitude, so
conflating them weakens both.
- *Leakage corrupts evaluation.* One model, scored on the contaminated test set and on the decontaminated one. The gap is the size of the lie. Large, unambiguous, no retraining involved.
- *Cleaning improves training.* An identical model trained three ways: on dirty data, on data with the same file count removed at random, and on Winnow-cleaned data, all scored on a verified-clean holdout. The random-removal arm is the control that matters, showing the gain comes from removing the correct files and not merely from removing files. The effect is small enough to be confused with seed variance, so every arm runs five seeds and reports mean ± sd, alongside a dose-response curve of accuracy against corruption rate. A trend beats a single before-and-after pair.

**Real-world validation.** Winnow runs against unmodified public datasets, and its
duplicate and leakage findings are compared against independently published
audits of those same datasets.

**Systems metrics.** Images per second per vCPU, wall-clock at 10k / 50k / 100k,
USD per 10k images, band index versus O(n²) candidate-generation time, and
Graviton versus x86 under identical build configuration with at least five
repetitions at p50 and p95. OpenCV's x86 vector paths are more mature than its
ARM ones, so a raw-throughput deficit alongside a price-performance win is the
expected and honestly reported result.

**Agent evaluation.** The agent is measured against the deterministic ranking,
not against nothing: accuracy recovered from its top 200 recommendations versus
the rule-based top 200 versus 200 random flags. A null result is reported as one.

**Documented failure cases.** Augmentation misread as duplication, legitimate
burst photography, datasets where blur is a feature, small-object and
uniform-texture annotations flagged as class outliers, screenshot datasets where
screen artifacts are the intended content, and synthetic imagery.

**Judge demo.** Public URL, no signup, scan already complete on arrival: clusters
with thumbnails, leakage surfaced first, the agent's plan with citations, and both
accuracy charts. One click runs a live scan on a smaller set, end to end, inside
five minutes.

---

## 7. Scope, sequencing, risk

**Week one is a dependency spike, before application code.** `img_hash` lives in
`opencv_contrib`, and the target is OpenCV 5 on ARM64, so a prebuilt distribution
for that combination is not assumed. The first task is proving that OpenCV 5 with
contrib builds and runs in an ARM64 Fargate container, with a Docker source build
as the fallback. This is the one dependency capable of disrupting the schedule,
so it gets resolved while there is time to react.

**Committed (MVP).** Near-duplicate detection, cross-split leakage, blur, EXIF
conflict, the single-task reduce stage, Graviton workers, the public dashboard,
and the leakage-corrupts-evaluation experiment. None of it requires label
ingestion.

**Declared upside, not commitments.** Annotation outlier detection, moiré,
exposure, the full dose-response study, agent-versus-baseline evaluation, and the
price-performance benchmark.

**Feature freeze 12 October,** leaving two weeks for evaluation runs, the report,
failure-case documentation and the video. Evidence is a scored deliverable, and
treating it as leftover time is the common way a strong build loses.

---

## 8. Use of the compute grant

For most entries compute runs the demo. Here compute is the result, because every
claim in §6 is a measurement that has to be paid for.

The dose-response study is four corruption rates across three arms at five seeds
each, so sixty training runs, and the seeds exist because one run cannot be
distinguished from noise. The scale study means full scans at 10k, 50k and 100k,
repeated for stable timings. The price-performance benchmark means running the
identical workload on x86 as well as Graviton, at least five repetitions each,
which is money spent on the architecture that will not ship so the comparison is
measured instead of asserted.

Judging runs 27 October to 9 November. The endpoint has to stay up across that
window and absorb scans judges trigger themselves. Entries frequently go dark
before they are judged.

I am a solo undergraduate entrant with no institutional or employer cloud budget.
The architecture is cost-disciplined by design, with read-in-place access, no
data duplication, no NAT gateway, tasks instead of always-on services, and one
reduce stage instead of a managed index, so a routine scan is cheap. The expense
concentrates entirely in repeated, statistically meaningful measurement.

**With the grant:** the full experimental matrix at the 100k ceiling, five seeds
per arm, the complete Graviton and x86 comparison, and an endpoint live
throughout judging. **Without it:** the ceiling drops to roughly 10k, seeds drop
to two, the x86 arm is cut, and the endpoint goes live only in short windows. The
project ships either way. The grant determines how strong the evidence behind it
is.

I am available for the required 21 September to 2 October check-in, and will
bring measured interim results instead of a status update.

---

## 9. Featured path

**Both.**

**COOL.** The per-image OpenCV 5 workload, which is the computational core, runs
on Graviton via ARM64 Fargate, with a controlled price-performance benchmark
against x86 in the final submission.

**Agentic Vision.** The Bedrock agent closes a real perceive, decide, act loop
over computed visual evidence. It decides which defects threaten accuracy and in
what order, then acts through quarantine manifests and targeted re-scans behind a
constrained, auditable tool interface, with its value measured against a
deterministic baseline instead of asserted.
