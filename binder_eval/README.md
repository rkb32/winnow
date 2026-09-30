# Binder overlap benchmark

An outside test of Winnow's train/test leak check on data it was not tuned on. The benchmark was built by Jesse Diaz: 300 synthetic Pokémon binder pages (240 train, 60 val, 1280x1280 JPEG, nine card slots each) in which every train/val overlap is planted and recorded in an answer key. Winnow's thresholds were calibrated on Imagenette and were not adjusted for this set.

**The page images are not in this repository.** The card art belongs to The Pokémon Company International, Nintendo, Creatures and GAME FREAK, and the benchmark was shared for private evaluation. Page names, card ids, counts and the flagged-pair list are fine to publish, so those are here. To rerun, get `binder-overlap-bench-v1.zip` from Jesse.

## Result

Winnow with its defaults (a pair is flagged at 25 keypoint matches that agree on one transform, or by the perceptual hash), all 14,400 train/val pairs:

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

## What it shows

- **Precision is the gap on template-heavy data.** Cards from different sets share frames, headers and symbols, the same way a stock template or a landmark does in the Imagenette audit (README, "Limitations"). Here the effect is much larger.
- **The embedding step filtered nothing.** Every page has the same dark background and grid, so all 14,400 pairs scored at least 0.82 in cosine similarity (Winnow's candidate bar is 0.80), and the keypoint check alone decided every pair.
- **The fixed-camera background check almost never applied** (12 of 14,400 pairs). It only steps in when the best match sits at the same pixels in both photos, and the false matches here are between cards in different slots.
- **Claude's review was off.** Winnow shows Claude the images only when a scan has 8 findings or fewer. With 773, the "Claude's call" column reads "not reviewed".
- **Not scored:** duplicates inside the training split, and the benchmark's own screening for reprinted art. Jesse notes his screen is not perfect, so a flag outside the key is worth a look; the strongest ones I checked were template matches.

## Files

- `pair_inliers.py`: keypoint matches for every train/val pair, in parallel (about 11 minutes on 5 workers).
- `score.py`: scores the pairs against `planted_pairs.csv` and writes the flagged-pair list.
- `winnow_flagged_pairs.csv`: the 773 flagged pairs: page A (train), page B (val), which check flagged it, matching keypoints, and Claude's call.

```
python binder_eval/pair_inliers.py path/to/binder-overlap-bench-v1 --out pair_inliers.csv
python binder_eval/score.py path/to/binder-overlap-bench-v1 pair_inliers.csv
```
