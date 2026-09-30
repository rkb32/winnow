# Binder overlap benchmark

An outside test of Winnow's train/test leak check on data it was not tuned on. The benchmark was built by Jesse Diaz: 300 synthetic Pokémon binder pages (240 train, 60 val, 1280x1280 JPEG, nine card slots each) in which every train/val overlap is planted and recorded in an answer key. Winnow's thresholds were calibrated on Imagenette and were not adjusted for this set, except for the shared-template filter described below, which this benchmark led to.

**The page images are not in this repository.** The card art belongs to The Pokémon Company International, Nintendo, Creatures and GAME FREAK, and the benchmark was shared for private evaluation. Page names, card ids, counts and the flagged-pair lists are fine to publish, so those are here. To rerun, get `binder-overlap-bench-v1.zip` from Jesse.

## Before: Winnow at its defaults

A pair is flagged at 25 keypoint matches that agree on one transform, or by the perceptual hash. All 14,400 train/val pairs:

| Planted overlap | Pairs | Flagged |
|---|---|---|
| One shared card (about 1/9 of the frame) | 12 | 9 |
| Two shared cards | 8 | 8 |
| Three shared cards | 6 | 6 |
| Re-shot (same cards, new lighting and compression) | 4 | 4 |
| Exact copy | 4 | 4 |
| **Planted pairs** | **34** | **31** |

The three missed pairs had 13, 19 and 23 matching keypoints, all with the shared card in the same slot.

It also flagged **742 of the 14,366 pairs that share nothing (5.2%)**, so 31 of the 773 flags are real overlaps. No pair of pages outside the key shares a card name. The strongest false alarm (68 matching keypoints) matches the printed "TRAINER" header, frame and corner symbols of two different Trainer cards, so the keypoint check is matching a shared printed template, not a shared card.

No keypoint bar separates the two groups:

| Smallest keypoint count that flags | Planted pairs found (of 34) | Flags outside the key |
|---|---|---|
| 25 (Winnow's default) | 31 | 742 |
| 30 | 29 | 516 |
| 40 | 29 | 294 |
| 50 | 26 | 83 |
| 60 | 22 | 10 |
| 80 | 21 | 0 |

## The fix: set aside matches made of a shared template

A printed template repeats across many different pages, so its keypoints match "everywhere". Winnow now checks each flagged pair's matching keypoints against 40 unrelated pages, treats any keypoint that also matches in 3 or more of them as template, and keeps the pair only if 20 template-free keypoints remain (`set_aside_templates` in `winnow/semantic.py`). A pair that fails is listed as a shared-template look-alike and not counted; nothing is deleted.

| Planted overlap | Pairs | Flagged after |
|---|---|---|
| One shared card | 12 | 8 |
| Two shared cards | 8 | 8 |
| Three shared cards | 6 | 6 |
| Re-shot | 4 | 4 |
| Exact copy | 4 | 4 |
| **Planted pairs** | **34** | **30** |

| | Flags outside the key | Set aside |
|---|---|---|
| Before | 742 | none |
| After | 31 | 712 (1 of them planted: train_044 / val_057, 27 keypoints, 19 template-free) |

Precision goes from 4% (31 of 773) to 49% (30 of 61). The 31 flags left outside the key have not been examined one by one.

**This result is in-sample.** The two settings (3 of 40 pages, 20 keypoints) were chosen with this benchmark's answer key in view. Two checks outside it: on the hand-verified Imagenette audit flags (`imagenette_exp/template_eval.py`), all 76 real leaks stay flagged and 8 of the 19 look-alikes are set aside (5 to 8, depending on which 40 photos are drawn); and the filter is off for sets under 100 photos. It has not been run on a set of near-identical frames, where a real copy's keypoints also match many other frames.

## Other findings

- **The embedding step filtered nothing.** Every page has the same dark background and grid, so all 14,400 pairs scored at least 0.82 in cosine similarity (Winnow's candidate bar is 0.80), and the keypoint check alone decided every pair.
- **The fixed-camera background check almost never applied** (12 of 14,400 pairs). It only steps in when the best match sits at the same pixels in both photos, and the false matches here are between cards in different slots.
- **Claude's review was off.** Winnow shows Claude the images only when a scan has 8 findings or fewer. With 773, the "Claude's call" column reads "not reviewed".
- **Not scored:** duplicates inside the training split, and the benchmark's own screening for reprinted art. Jesse notes his screen is not perfect, so a flag outside the key is worth a look; the strongest ones I checked were template matches.

## Files

- `pair_inliers.py`: keypoint matches for every train/val pair, in parallel (about 11 minutes on 5 workers).
- `score.py`: scores the pairs against `planted_pairs.csv`, writes the flagged-pair list, and with `--templates-out` also scores with shared templates set aside.
- `winnow_flagged_pairs.csv`: the 773 pairs Winnow flagged before the filter: page A (train), page B (val), which check flagged it, matching keypoints, and Claude's call.
- `winnow_flagged_pairs_templates_set_aside.csv`: the same 773 pairs after it, with 712 marked "shared template (set aside)" and their template-free keypoint count.

```
python binder_eval/pair_inliers.py path/to/binder-overlap-bench-v1 --out pair_inliers.csv
python binder_eval/score.py path/to/binder-overlap-bench-v1 pair_inliers.csv --templates-out winnow_flagged_pairs_templates_set_aside.csv
```
