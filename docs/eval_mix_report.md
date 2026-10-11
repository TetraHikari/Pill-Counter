# Evaluation: mixed-data counter thai_mix_s1024 (10 Oct 2026)

> Working record from the project, lightly edited for this repository. Some paths it mentions are not
> included here: other model checkpoints and threshold files, `backups/`, `drive/`, `results/app_uploads/`
> (private photos), `results/s20_diag/`, `results/own_large_piles/`, `results/eval_other_pills_v2/` and the
> internal run notes. See the README for what is included.

Evaluation of the counter trained on our photos plus public multi-drug photos. The test set was not run. The app model, `src/api.py`
and the app's thresholds files were not changed, and nothing was trained.

## Model file check
`thai_mix_s1024.pt` (19.2 MB) was copied from `~/Downloads` to `drive/` and `results/models/`. It loads normally. It has one class
(`pill`), imgsz 1024 and 60 epochs. The run name is `mix_s1024_stage2`, initialised from `mix_s1024_stage1/weights/best.pt` and trained on
`mixed.yaml`, which matches the recipe in `notebooks/train_counter_mix_colab.ipynb`.

## 1. Validation (60 photos; sessions s06 s13 s18 s23 s26)
`tools/compare_counters.py` gained `--rule photo|session|both`. The default is `photo`, which is the old rule and behaves the same as before.
`session` chooses the setting by the mean of the five per-session MAEs, so each session counts equally. s26 alone is 30 of the
60 photos. Both rules search the same grid: IoU 0.4–0.7 × conf 0.10–0.85, tie-broken by exact rate and then by conf closest to 0.5.
Run: `.venv/bin/python tools/compare_counters.py --rule both thai_v2 thai_v3_m1024 thai_v4m_m1024 thai_mix_s1024`.

### Chosen settings and overall scores
| Model | Rule | conf | IoU | photo-weighted exact / MAE | session-weighted exact / MAE |
|---|---|---|---|---|---|
| thai_v2 | photo | 0.55 | 0.4 | 75% / 0.28 | 81% / 0.20 |
| thai_v2 | session | 0.50 | 0.4 | 73% / 0.30 | 85% / 0.17 |
| thai_v3_m1024 | photo | 0.40 | 0.4 | 85% / 0.20 | 83% / 0.23 |
| thai_v3_m1024 | session | 0.20 | 0.4 | 80% / 0.20 | 79% / 0.21 |
| thai_v4m_m1024 | photo = session | 0.50 | 0.4 | 82% / 0.30 | 82% / 0.27 |
| **thai_mix_s1024** | photo = session | **0.50** | **0.4** | **92% / 0.12** | **91% / 0.13** |

### Per session, exact % / MAE (true count, photos)
| Session | thai_v2 photo (0.55/0.4) | thai_v2 session (0.50/0.4) | m1024 photo (0.40/0.4) | m1024 session (0.20/0.4) | v4m (0.50/0.4) | **mix_s1024 (0.50/0.4)** |
|---|---|---|---|---|---|---|
| s06 (39, 5) | 100% / 0.0 | 100% / 0.0 | 100% / 0.0 | 100% / 0.0 | 80% / 0.2 | 100% / 0.0 |
| s13 (25, 10) | 60% / 0.4 | 90% / 0.1 | 60% / 0.7 | 80% / 0.2 | 80% / 0.3 | 80% / 0.4 |
| s18 (14, 10) | 100% / 0.0 | 100% / 0.0 | 80% / 0.2 | 70% / 0.3 | 90% / 0.1 | 100% / 0.0 |
| s23 (40, 5) | 80% / 0.2 | 80% / 0.2 | 80% / 0.2 | 60% / 0.4 | 80% / 0.4 | 80% / 0.2 |
| s26 (90, crowded tray, 30) | 67% / 0.4 | 53% / 0.5 | 93% / 0.1 | 83% / 0.2 | 80% / 0.4 | 93% / 0.1 |

The photo-rule rows reproduce `docs/eval_v4m_report.md` exactly. Files written:
`results/thresholds_<model>_compare.json` (photo rule, same values as before; it now also stores session-weighted
scores and `rule`) and `results/thresholds_<model>_compare_session.json` (session rule) for all four models.

## 2. Mixed tray s20 (evaluation only; 20 photos, true 82 = 20 tablet / 40 caplet / 22 mini_tablet)
`thresholds_thai_mix_s1024_compare_session.json` was copied to `results/thresholds_thai_mix_s1024.json` (conf 0.50, IoU 0.4).
Then `eval_mixed.py --counter thai_mix_s1024.pt --thresholds thai_mix_s1024 --trays s20` was run with classifier `pill_type.pt`. The
other rows come from `docs/eval_v4m_report.md` (app thresholds) and were not re-run.

| Counter | TOTAL exact / miss | tablet | caplet | mini_tablet | all three forms exact |
|---|---|---|---|---|---|
| thai_v2 (0.50 / 0.50) | 60% / 0.8 | 90% / 0.1 | 60% / 0.8 | 100% / 0.0 | 55% |
| thai_v3_m1024 (0.40 / 0.40) | 50% / 0.8 | 90% / 0.1 | 60% / 0.7 | 100% / 0.0 | 50% |
| thai_v4m_m1024 (0.50 / 0.40) | 20% / 2.4 | 45% / 1.1 | 25% / 1.6 | 100% / 0.0 | 10% |
| **thai_mix_s1024 (0.50 / 0.40)** | **30% / 2.4** | 95% / 0.1 | 30% / 2.4 | 100% / 0.0 | **30%** |

Per-photo total counts for thai_mix_s1024 on s20 (true 82):
81 69 71 81 80 82 77 78 81 80 82 81 78 82 82 82 81 81 82 81

thai_mix_s1024 **under-counts** s20. It is never above 82, it gets 6 of 20 photos exact, and two photos are far off (69 and 71, −13 and −11).
Those two photos account for 24 of the 48 missed pills. Nearly all of the error falls on caplets (tablets 95% exact).
thai_v4m_m1024 had the same average miss but mostly over-counted.

## 3. Other medicines (Colab, `results/eval_other_pills_v2/`)
These come from `notebooks/eval_other_pills_colab.ipynb`, run on Colab and unzipped from the Drive download into
`results/eval_other_pills_v2/` (`report.md`, `per_image.csv`, `confs.npz`, `worst/`). Every model uses the thresholds tuned on our val photos:
thai_v2 0.50 / 0.50, thai_v3_m1024 0.40 / 0.40, thai_v4m_m1024 0.50 / 0.40 and thai_mix_s1024 0.50 / 0.40. All runs use imgsz 1024. Error is
predicted − true, and a positive bias means over-counting.

### Pill-Counter v2, held-out test part (187 images, 642 pills, 1–18 per image; no model trained or chose on it)
| Model | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|
| thai_v2 | 25.1 | 1.9 | −0.3 | 20.9 | 54.0 | 63.1 | 30 |
| thai_v3_m1024 | 11.2 | 3.3 | −0.4 | 19.8 | 69.0 | 119.6 | 101 |
| thai_v4m_m1024 | 42.2 | 1.8 | +0.9 | 32.1 | 25.7 | 68.4 | 44 |
| **thai_mix_s1024** | **62.0** | **0.6** | −0.4 | 5.9 | 32.1 | 20.2 | 4 |

179 of the 187 images are "numbered drug ids" and 8 are brown_capsule. For thai_mix_s1024, the held-out MAE at other
cut-offs (app IoU) is conf 0.3 → 0.54, 0.4 → 0.49, 0.5 → 0.58 and 0.6 → 0.86. Its remaining errors lean to under-counting
(32% of images under, 6% over).

Caveat: no model saw these 187 images, but they come from the same Pill-Counter v2 source as the part thai_mix_s1024 was
trained on. For thai_mix_s1024 this set is an unseen sample of a familiar source. For the other three it is a new source.

### Other sets (marked where a model saw them)
| Set | thai_v2 | thai_v3_m1024 | thai_v4m_m1024 | thai_mix_s1024 |
|---|---|---|---|---|
| Pill-Counter v2, all 1,255 (exact % / MAE) | 35.8 / 1.6 | 11.0 / 3.7 | 50.5 / 1.7 | 65.9 / 0.5 (**seen**: 70% train, 15% val) |
| v4 valid, 182 (exact % / MAE) | 65.9 / 1.3 (seen-ish) | 32.4 / 11.3 | 38.5 / 2.5 (seen-ish) | 92.9 / 0.1 (**seen**: validation) |
| pill count v1 trays, 112 frames, 10–175 pills (exact % / MAE / bias) | 0.0 / 40.8 / −39.7 | 0.9 / 20.7 / −19.5 | 0.0 / 27.7 / −26.6 | 0.0 / 42.9 / −41.8 |

The v1 tray frames are unseen by every model. thai_mix_s1024 has the highest MAE there, and every model under-counts almost every frame.
thai_v3_m1024 is the least wrong in every video and in every pile bin above 20. The 1–20 bin is a single frame, which every model over-counts by 60–65. Lower cut-offs reduce but do not remove this error.
thai_mix_s1024's v1 MAE is 26.4 at conf 0.1, against 12.0 for thai_v3_m1024 at the same cut-off.

## 4. Speed (this Mac: Apple M1 Max, `device()` = CPU, imgsz 1024)
Same method as `docs/eval_v4m_report.md`: 10 val photos (2 per session), 3 warm-up predictions, then 3 passes (30 timed predictions
per model). Detector only (`predict`, including pre- and post-processing), at compare thresholds.

| Model | median | mean | min–max |
|---|---|---|---|
| thai_v2 (11s) | 194 ms | 194 ms | 183–204 ms |
| thai_v3_m1024 (11m) | 377 ms | 379 ms | 367–407 ms |
| thai_v4m_m1024 (11m) | 376 ms | 379 ms | 369–431 ms |
| thai_mix_s1024 (11s) | 197 ms | 198 ms | 186–208 ms |

All four are about 5% faster than in the v4m report, presumably because the machine was less loaded. The ratios are unchanged.
thai_mix_s1024 runs at thai_v2's speed, about 1.9× faster than the m models.

## 5. Reading
- **Validation:** thai_mix_s1024 has the best overall numbers of the four under both weightings (92% / 0.12 photo-weighted,
  91% / 0.13 session-weighted). Both rules pick the same setting for it, conf 0.50 / IoU 0.4. It matches thai_v3_m1024 on the crowded
  tray s26 (93% / 0.1) and thai_v2 on s06 and s18 (100%). Its weakest session is s13 (80% / 0.4).
- **Rule choice:** the session-weighted rule changes the setting only for thai_v2 (0.55 → 0.50) and thai_v3_m1024
  (0.40 → 0.20). In both cases it trades s26 accuracy for small-session accuracy. Under that rule thai_v2 has the best s13 of any row
  (90% / 0.1). The rule makes no difference for v4m or mix_s1024.
- **Sample size:** s06 and s23 have 5 photos each and s13 and s18 have 10, so one photo moves those rates by 10–20 points.
  The gap between thai_mix_s1024 and the next best on validation (photo-weighted 92% vs 85%) is 4 photos out of 60.
- **Mixed tray s20:** thai_mix_s1024 is worse than thai_v2 and thai_v3_m1024 (30% total exact vs 60% / 50%; all three
  forms 30% vs 55% / 50%) and better than v4m on exact rates. Its errors are under-counts on caplets, concentrated in two photos.
- **Speed:** the same as thai_v2.
- **Other medicines, held-out Pill-Counter v2 part:** thai_mix_s1024 is clearly ahead (62% exact / MAE 0.6 vs 42% / 1.8
  for v4m, 25% / 1.9 for thai_v2 and 11% / 3.3 for thai_v3_m1024). This set comes from the same source as part of its training
  data and is a new source for the other three. That makes it a test of whether the extra data generalises within that source,
  not a test on fully new data.
- **Other medicines, unseen v1 tray frames (big piles, 640×480 video):** thai_mix_s1024 does not improve on thai_v2 there
  (MAE 42.9 vs 40.8). Both are about twice thai_v3_m1024's 20.7, and every model under-counts these frames.

## Files written
- `tools/compare_counters.py` (new `--rule` option; default behaviour unchanged)
- `results/thresholds_{thai_v2,thai_v3_m1024,thai_v4m_m1024,thai_mix_s1024}_compare.json` (rewritten with the same values as before plus extra
  fields), `results/thresholds_*_compare_session.json` (new)
- `results/thresholds_thai_mix_s1024.json` (copy, for `eval_mixed.py`; not used by the app)
- `notebooks/make_eval_other_pills_nb.py`, `notebooks/eval_other_pills_colab.ipynb` (thai_mix_s1024 thresholds set,
  thai_mix_m1024 commented out until tuned)
- `results/eval_other_pills_v2/` (unzipped from the Colab download)
- This report. The speed script was a one-off and is not included.
- `drive/thai_mix_s1024.pt`, `results/models/thai_mix_s1024.pt`

## s20 diagnosis (follow-up)
Diagnosis only. Nothing was trained, the test set was not run, and the app's model and thresholds files were not changed.
Script and outputs are in `results/s20_diag/`: `s20_diag.py`, `per_pill.csv`, `per_photo.csv` and the four drawings.

### Main finding: the validation tuning did not score photos the way the app does
`compare_counters.py` passes all 60 val photos to `predict` as one list. 8 of them (s26_IMG_3438/39/46–51) are
5712×4284 and the rest are 4032×3024. When the list mixes image sizes, Ultralytics 8.4.173 letterboxes **every** photo to a padded
1024×1024 square, even with `batch=1`. The app (`src/api.py:57`), `eval_mixed.py` and the Colab notebook predict one
photo per call, and that gives the rectangular 1024×768 letterbox. On the same photo and thresholds the counts differ:
s26_IMG_3453 gets 90 in the list and 87 alone. So **every validation number and compare threshold in this report, in
`docs/eval_v4m_report.md` and in PROGRESS.md was measured in a mode the app does not use.** `eval_mixed.py` (s20), the Colab other-medicine
run and the speed runs are app-style and are not affected.

I re-ran the same grid and rule one photo per call (scratch copy of `compare_counters.py`; the repo file is unchanged).
The chosen thresholds were saved as `results/thresholds_<model>_single.json` and copied into `results/s20_diag/`. Both weightings pick the same
setting for every model.

| Model | Square-mode choice → app-style score at it | App-style choice | App-style val at it: photo-w / session-w exact, MAE |
|---|---|---|---|
| thai_v2 | 0.55 / 0.4 → 55% / 0.70 (app file 0.50 / 0.50 → 68% / 0.37) | 0.55 / 0.6 | 77% / 0.30, 87% / 0.17 |
| thai_v3_m1024 (app now) | 0.40 / 0.40 → **80% / 0.25** (reported 85% / 0.20) | 0.25 / 0.5 | 87% / 0.15, 85% / 0.19 |
| thai_v4m_m1024 | 0.50 / 0.4 → 60% / 0.55 | 0.60 / 0.5 | 82% / 0.25, 82% / 0.25 |
| **thai_mix_s1024** | 0.50 / 0.4 → **67% / 0.48** (reported 92% / 0.12) | **0.55 / 0.6** | **93% / 0.10, 95% / 0.09** |

App-style per session at the app-style choice (exact / MAE):

| Session | thai_v2 0.55/0.6 | m1024 0.25/0.5 | v4m 0.60/0.5 | mix_s1024 0.55/0.6 |
|---|---|---|---|---|
| s06 | 100% / 0.0 | 100% / 0.0 | 100% / 0.0 | 100% / 0.0 |
| s13 | 70% / 0.4 | 100% / 0.0 | 80% / 0.3 | 80% / 0.4 |
| s18 | 100% / 0.0 | 80% / 0.2 | 90% / 0.1 | 100% / 0.0 |
| s23 | 100% / 0.0 | 60% / 0.6 | 60% / 0.6 | 100% / 0.0 |
| s26 | 63% / 0.5 | 87% / 0.1 | 80% / 0.3 | 93% / 0.1 |

### What the s20 misses are (thai_mix_s1024 at 0.50 / 0.4)
Each counted box was matched to the s20 label boxes (82 per photo, from Roboflow) at IoU ≥ 0.3. The model's boxes are about 25 px
looser than the labels, so IoU 0.5 dropped true hits on small pills. The form of each pill comes from `pill_type.pt` on its label
box (it finds exactly 20 / 40 / 22 per photo). With this matching, missed − extra equals true − counted on every photo.

- **All 49 missed pills on s20 are caplets.** No tablet or mini tablet was missed. The two worst photos account for 24 of them
  (s20_IMG_3343: 13, s20_IMG_3344: 11). The other 25 are spread over 13 photos (1–6 each), so it is not a single-photo event.
- **Cause: 44 of 49 are NMS-suppressed.** A box sits on the missed caplet at IoU 0.7, but it is removed at IoU 0.4 by a neighbouring
  caplet's box. Two were below the conf cut-off and 3 were never detected. For comparison, thai_v2 (0.50 / 0.50) missed 19 caplets,
  15 of them below conf.
- **Layout:** in the drawings, every miss in s20_IMG_3343 and s20_IMG_3344 sits inside the stacked caplet pile. These are among the first
  5 photos, where the three forms lie in separate piles, so the caplets form one dense heap. None are at the image edge, in the glare spot or isolated.
  The tablet and mini-tablet piles are fully counted. My automatic touching/overlap tag uses axis-aligned boxes, which
  overlap for any diagonal caplet, so it does not separate the cases; the visual check is what supports this description.
- **Same pattern on the beige trays.** At 0.50 / 0.4 all of mix_s1024's misses on s26 (val, 19), s25 (9), s27 (3) and s28 (2) are caplets
  suppressed by NMS. s25, s27 and s28 are training trays.

### Same photos at the app-style choice (0.55 / 0.6, picked on validation only, not on s20)
| Set | 0.50 / 0.4: exact / MAE / net | 0.55 / 0.6: exact / MAE / net |
|---|---|---|
| s20 (eval only), total | 30% / 2.40 / −48 | **95% / 0.05** (19 of 20 photos at 82, one at 81) |
| s26 (val) | 53% / 0.63 / −17 | 93% / 0.07 / +2 |
| s25 (train) | 63% / 0.37 / −7 | 97% / 0.03 / +1 |
| s27 (train) | 80% / 0.20 / −3 | 100% / 0.00 / 0 |
| s28 (train) | 91% / 0.09 / −2 | 100% / 0.00 / 0 |

s20 with the full counter + classifier (`eval_mixed.py`), each model at its app-style choice:

| Counter | TOTAL exact / miss | tablet | caplet | mini_tablet | all three forms exact |
|---|---|---|---|---|---|
| thai_v2 (0.55 / 0.6) | 40% / 1.3 | 90% / 0.1 | 45% / 1.3 | 100% / 0.0 | 40% |
| thai_v3_m1024 (0.25 / 0.5) | 75% / 0.3 | 90% / 0.1 | 85% / 0.2 | 95% / 0.1 | 75% |
| thai_v4m_m1024 (0.60 / 0.5) | 55% / 0.8 | 50% / 0.7 | 80% / 0.3 | 100% / 0.0 | 50% |
| **thai_mix_s1024 (0.55 / 0.6)** | **95% / 0.1** | 95% / 0.1 | 90% / 0.1 | 100% / 0.0 | **90%** |

Drawings (left: thai_mix_s1024 at 0.50 / 0.4, with boxes at conf 0.25–0.50 in orange; right: thai_v2 at 0.50 / 0.50. Red marks a missed labelled pill and
magenta marks a pill only one of the two models counted): `results/s20_diag/s20_IMG_3343_mix_vs_v2.jpg`, `s20_IMG_3344_mix_vs_v2.jpg`
(worst), and `s20_IMG_3342_mix_vs_v2.jpg` and `s20_IMG_3347_mix_vs_v2.jpg` (typical, 81 and 82).

### Reading
- The s20 under-count is **systematic, not a one-off**. It comes from NMS at IoU 0.4 removing boxes on overlapping or stacked caplets,
  it appears on both surfaces (floor tile and beige tray), and it is worst where caplets are heaped. It is not lighting, edges or the
  detector failing to see the pills: 44 of 49 missed caplets had a box before NMS.
- The IoU 0.4 setting itself came from the square-letterbox tuning. Scored the way the app runs, validation picks IoU 0.6 for
  thai_mix_s1024, and at that setting the s20 caplet problem largely disappears (95% total exact). That setting was chosen on validation, not on s20.
- The letterbox issue affects all models' earlier validation numbers and thresholds, including the app's thai_v3_m1024 0.40 / 0.40 (app-style val
  80% / 0.25, against 85% / 0.20 reported). `compare_counters.py` was not changed and no app file was touched. Fixing the tuning
  (one `predict` call per photo) and any threshold change were left to a separate decision.
- The Colab other-medicine numbers for thai_mix_s1024 (section 3) were at 0.50 / 0.4. At conf 0.55 they could be re-cut from
  `confs.npz`, but IoU 0.6 needs a new run.

## Fix (scoring one photo per call)
After that decision, the app was switched to thai_mix_s1024 at 0.55 / 0.6 (`results/thresholds_thai_mix_s1024_single.json`).
This step changed only the evaluation code:

- `tools/compare_counters.py`: one `predict` call per photo (was one list for all 60). Options (`--rule photo|session|both`) and
  output files are unchanged. Re-run `--rule both` for the four models: it reproduces every `*_single.json` choice and score
  exactly, with no differences. Both rules pick the same setting for every model.

  | Model | conf / IoU | photo-w exact / MAE | session-w exact / MAE |
  |---|---|---|---|
  | thai_v2 | 0.55 / 0.6 | 77% / 0.30 | 87% / 0.17 |
  | thai_v3_m1024 | 0.25 / 0.5 | 87% / 0.15 | 85% / 0.19 |
  | thai_v4m_m1024 | 0.60 / 0.5 | 82% / 0.25 | 82% / 0.25 |
  | thai_mix_s1024 | 0.55 / 0.6 | 93% / 0.10 | 95% / 0.09 |

  This overwrote `results/thresholds_<model>_compare.json` and `_compare_session.json` with these app-style values. The
  square-mode values are in sections 1 and 2 above. The app reads `thresholds_thai_mix_s1024_single.json`, which was not touched.
- `tools/evaluate.py`: `box_confs` predicts one photo per call. `report --split val` runs and its numbers agree with the
  app-style ones: L5 thai_v2 (0.50 / 0.50) 68% / 0.37, and the "L5 app counter" row thai_v3_m1024 (0.40 / 0.40) 80% / 0.25. That row is still
  thai_v3_m1024 because `LEVELS` in `evaluate.py` was not changed. The test set was not run. The previous `results/val_results.md` and
  `val_predictions.csv` were backed up to `backups/eval_fix_2026-10-10/` before being overwritten.
- `notebooks/make_eval_other_pills_nb.py` and the notebook: every model at its app-style thresholds, plus `thai_v2_appfile` (0.50 / 0.50).
  Output goes to `eval_other_pills_v3/`. Results are in the next section.

## Other medicines at app-style thresholds (Colab v3, `results/eval_other_pills_v3/`)
Each model ran at its app-style thresholds. `thai_v2_appfile` is thai_v2 at its old app file 0.50 / 0.50. Its rows reproduce the v2 run's
thai_v2 rows exactly (held-out 25.1% / 1.9, v1 trays MAE 40.8), which checks the two runs against each other.

### Pill-Counter v2, held-out test part (187 images, 642 pills, 1–18 per image)
| Model (conf / IoU) | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|
| thai_v2 (0.55 / 0.6) | 22.5 | 1.8 | −0.7 | 16.6 | 61.0 | 59.5 | 24 |
| thai_v2_appfile (0.50 / 0.50) | 25.1 | 1.9 | −0.3 | 20.9 | 54.0 | 63.1 | 30 |
| thai_v3_m1024 (0.25 / 0.5) | 16.6 | 3.6 | +0.9 | 29.4 | 54.0 | 140.6 | 115 |
| thai_v4m_m1024 (0.60 / 0.5) | 36.9 | 1.6 | +0.1 | 21.4 | 41.7 | 55.8 | 42 |
| **thai_mix_s1024 (0.55 / 0.6)** | **57.8** | **0.7** | −0.5 | 4.3 | 38.0 | 21.5 | 5 |

thai_mix_s1024 was 62.0% / 0.6 at the old 0.50 / 0.4. The same caveat applies: these images are unseen, but they come from a source it trained on.

### pill count v1 tray video frames (112 frames, 10–175 pills, 640×480; unseen by all)
| Model | exact % | MAE | bias | MAE 21–50 | MAE 51–100 | MAE 100+ |
|---|---|---|---|---|---|---|
| thai_v2 (0.55 / 0.6) | 0.0 | 41.9 | −40.7 | 26.4 | 30.0 | 72.2 |
| thai_v2_appfile | 0.0 | 40.8 | −39.7 | 26.0 | 29.0 | 70.4 |
| **thai_v3_m1024 (0.25 / 0.5)** | 4.5 | **16.4** | −15.2 | 12.5 | 9.7 | 28.6 |
| thai_v4m_m1024 (0.60 / 0.5) | 0.0 | 31.3 | −30.2 | 21.0 | 20.9 | 55.0 |
| thai_mix_s1024 (0.55 / 0.6) | 0.0 | **47.8** | −46.8 | 29.3 | 37.5 | 78.3 |

thai_mix_s1024 under-counts the v1 frames the most of the five (it was 42.9 at 0.50 / 0.4). Its MAE at lower cut-offs (IoU 0.6) is
24.6 at conf 0.1 and 36.3 at 0.4, still above thai_v3_m1024 at the same cut-offs (11.1, 20.6). v4 valid is seen by thai_mix_s1024 (95.1% / 0.1).

## Own large piles (app time-test uploads, `results/own_large_piles/`)
These are six photos from the app time test (`results/app_uploads/`, 4032×3024, mixed tablets, caplets and mini tablets on the beige tray / floor tile), with one
`predict` per photo as in the app. They have **no hand labels**: "true" is the pile size laid out by hand (the 60-pill photo is confirmed
as 60). Script: `results/own_large_piles/own_large_piles.py`, counts in `counts.csv`.

| Photo | true | mix_s1024 0.55/0.6 | m1024 0.25/0.5 | m1024 0.40/0.40 | thai_v2 0.50/0.50 |
|---|---|---|---|---|---|
| 20261010-142612 | 10 | 10 | 10 | 10 | 10 |
| 20261010-142708 | 50 | 50 | 50 | 50 | 50 |
| 20261010-142651 | 60 | 60 | 60 | 60 | 60 |
| 20261010-142748 | 100 | 100 | 100 | 100 | 101 |
| 20261010-142218 | 100 | 99 | 100 | 100 | 99 |
| 20261010-142206 | 100 | 100 | 101 | 101 | 100 |

The m1024 0.40/0.40 counts match what the app reported at the time (the upload `.json` files).
thai_mix_s1024 misses one pill in the six photos (142218: 99). Its missing pill had a box at conf 0.25–0.55, so it was below the cut-off,
not NMS-suppressed. In the drawing (`results/own_large_piles/20261010-142218_mix_dots.jpg`) it is a lone yellow mini tablet on the tray
rim, away from the pile. Raising NMS IoU from 0.6 to 0.7 adds no box on any of the three 100-pill photos. Drawings for the other two 100-pill
photos are `…142748_mix_dots.jpg` and `…142206_mix_dots.jpg`.

Reading: on our own 100-pill mixed piles (spread out, few heaps), every model is within ±1, so these photos do not separate
the models. The large v1 under-count of thai_mix_s1024 does not show up on our own photos. The v1 frames differ from them in camera
(640×480 video frames, much smaller pills in pixels) as well as pile size, and this check cannot tell which of the two causes the gap.

## Test set, run 2 (disclosed)
**This is the second use of the test set.** It was planned and confirmed by the author before it ran. The final model was
fixed beforehand as thai_mix_s1024 at 0.55 / 0.6. The command `.venv/bin/python tools/evaluate.py report --split test --final --again` was run once, at 20:50.
`results/test_log.txt` now has two lines (14:40 and 20:50). Run 1 (`test_results.md`, `test_predictions.csv`, `test_log.txt`) is backed up
in `backups/test_run1_2026-10-10/`. The new files are `results/test_results.md` and `test_predictions.csv`. The s20 result for this model and setting is in the s20
diagnosis section (95% total exact, 90% all forms), so it was not re-run.

All 45 test photos are 4032×3024, so run 1's list-based scoring was not affected by the letterbox issue. The rows shared by both runs
(L1–L4, thai_v2) are identical.

### All conditions (45 photos, 6 sessions; 95% CI by resampling sessions)
| Level | Exact [95% CI] | MAE [95% CI] | Rel. error | Bias |
|---|---|---|---|---|
| L1 mean train count | 0% [0, 0] | 37.33 [26.66, 45.55] | 232.4% | +37.33 |
| L2 classical watershed | 4% [0, 8] | 9.09 [4.38, 14.05] | 38.2% | −7.89 |
| L3 pretrained YOLO (COCO), 0.25 / 0.7 | 0% [0, 0] | 23.18 [14.09, 33.94] | 89.3% | −23.18 |
| L4 v4 pill model, 0.50 / 0.4 | 82% [57, 95] | 0.69 [0.05, 2.10] | 2.0% | +0.69 |
| L5 thai_v2, 0.50 / 0.50 | **93%** [91, 97] | **0.07** [0.03, 0.09] | 0.4% | +0.02 |
| L5 thai_v3_m1024, 0.25 / 0.5 | 82% [71, 93] | 0.20 [0.07, 0.31] | 1.5% | +0.20 |
| **L5 app counter thai_mix_s1024, 0.55 / 0.6** | 89% [78, 98] | 0.13 [0.02, 0.27] | 0.7% | −0.13 |

Seen conditions (35 photos): thai_mix_s1024 91% / 0.09, thai_v2 91% / 0.09, m1024 80% / 0.23.
Unseen conditions (s03, s04 blue tablets on brown cloth, 10 photos): thai_mix_s1024 80% / 0.30, thai_v2 100% / 0.00, m1024 90% / 0.10, L4 100% / 0.00.

### Per session, predicted counts per photo
| Session | True | thai_v2 0.50/0.50 | thai_v3_m1024 0.25/0.5 | thai_mix_s1024 0.55/0.6 |
|---|---|---|---|---|
| s03 brown_cloth (unseen) | 31 | 31 31 31 31 31 | 32 31 31 31 31 | 31 **29** 31 31 31 |
| s04 brown_cloth (unseen) | 35 | 35 35 35 35 35 | 35 35 35 35 35 | 35 35 35 35 **34** |
| s08 granite | 46 | 46 46 46 46 46 | 46 46 46 46 46 | 46 46 46 46 46 |
| s14 floor_tile | 15 | 15 15 15 15 15 15 15 15 15 14 | 15 15 15 15 16 15 15 15 15 17 | 15 15 15 15 **14** 15 15 **14** 15 **14** |
| s19 floor_tile | 10 | 10 10 10 10 11 10 10 10 10 10 | 11 11 10 11 11 10 10 10 10 10 | 10 10 10 10 10 10 10 10 10 10 |
| s24 floor_tile | 30 | 31 30 30 30 30 30 30 30 30 30 | 30 30 30 30 31 30 30 30 30 30 | 30 30 30 30 30 30 30 30 30 30 |

### Compared with run 1 (14:40, app counter then thai_v3_m1024 at 0.40 / 0.40)
| | Run 1 | Run 2 |
|---|---|---|
| App counter | thai_v3_m1024 0.40 / 0.40: 84% [76, 97] / 0.18 | thai_mix_s1024 0.55 / 0.6: 89% [78, 98] / 0.13 |
| thai_v3_m1024 | 0.40 / 0.40: 84% / 0.18 | 0.25 / 0.5: 82% / 0.20 |
| thai_v2 0.50 / 0.50 | 93% / 0.07 | 93% / 0.07 (same predictions) |

Reading:
- thai_mix_s1024 is between thai_v2 and thai_v3_m1024 on the test set: 89% exact against 93% and 82–84%. The confidence intervals of all three overlap.
- Its 5 misses are all under-counts by 1–2: s03 one photo −2, s04 one −1, s14 three −1. thai_v2 has 3 misses, all ±1.
- thai_mix_s1024 is exact on every photo of s08, s19 and s24. Its misses are on the unseen brown-cloth sessions (2 of 10 photos) and on caplets on floor tile (s14, 3 of 10).
- Moving thai_v3_m1024 from 0.40 / 0.40 to its app-style 0.25 / 0.5 changed it little on test (84% → 82%, MAE 0.18 → 0.20).
