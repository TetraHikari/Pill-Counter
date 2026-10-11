# Evaluation: counter thai_v4m_m1024 (10 Oct 2026)

> Working record from the project, lightly edited for this repository. Some paths it mentions are not
> included here: other model checkpoints and threshold files, `backups/`, `drive/`, `results/app_uploads/`
> (private photos), `results/s20_diag/`, `results/own_large_piles/`, `results/eval_other_pills_v2/` and the
> internal run notes. See the README for what is included.

Evaluation of the YOLO11m counter trained from the v4 public-pill stage. Test set not run; app model, `src/api.py`
and the app's thresholds files unchanged; nothing trained.

## Model file check
`results/models/thai_v4m_m1024.pt` (40.6 MB) loads normally: one class (`pill`), `imgsz` 1024, 60 epochs,
run name `v4m_m1024_stage2`, initialised from `v4m_m1024_stage1/weights/best.pt` (the v4 public-pill stage),
trained on `real.yaml`. That matches the two-stage recipe in `notebooks/train_counter_v4m_colab.ipynb`. Nothing looked wrong.

## 1. Validation (60 photos, sessions s06 s13 s18 s23 s26)
`.venv/bin/python tools/compare_counters.py thai_v2 thai_v3_m1024 thai_v4m_m1024`. Each model is tuned the
same way (lowest mean miss over IoU 0.4–0.7 × conf 0.10–0.85).

| Model | conf | IoU | ALL exact | ALL MAE |
|---|---|---|---|---|
| thai_v2 (YOLO11s, COCO → v4 → ours) | 0.55 | 0.4 | 75% | 0.28 |
| thai_v3_m1024 (YOLO11m, COCO → ours) | 0.40 | 0.4 | **85%** | **0.20** |
| thai_v4m_m1024 (YOLO11m, COCO → v4 → ours) | 0.50 | 0.4 | 82% | 0.30 |

Per session, exact % / MAE (true count and number of photos in brackets):

| Session | thai_v2 | thai_v3_m1024 | thai_v4m_m1024 |
|---|---|---|---|
| s06 (39 pills, 5 photos) | 100% / 0.0 | 100% / 0.0 | 80% / 0.2 |
| s13 (25, 10) | 60% / 0.4 | 60% / 0.7 | **80% / 0.3** |
| s18 (14, 10) | 100% / 0.0 | 80% / 0.2 | 90% / 0.1 |
| s23 (40, 5) | 80% / 0.2 | 80% / 0.2 | 80% / 0.4 |
| s26 (90, crowded tray, 30) | 67% / 0.4 | **93% / 0.1** | 80% / 0.4 |

The thai_v2 and thai_v3_m1024 rows match the earlier validation numbers. Note that thai_v2's compare
setting (0.55 / 0.4) differs from its app setting (0.50 / 0.50); the app file was not touched.
Saved `results/thresholds_thai_v4m_m1024_compare.json` (conf 0.50, IoU 0.4, imgsz 1024).

## 2. Mixed tray s20 (evaluation only; 20 photos, true 82 = 20 tablet / 40 caplet / 22 mini_tablet)
Copied `thresholds_thai_v4m_m1024_compare.json` → `results/thresholds_thai_v4m_m1024.json`, then ran
`eval_mixed.py --trays s20`. Classifier `pill_type.pt` for all. I re-ran the two references with their app
thresholds, and both matched the numbers from the earlier evaluation.

| Counter | TOTAL exact / miss | tablet | caplet | mini_tablet | all three forms exact |
|---|---|---|---|---|---|
| thai_v2 (0.50 / 0.50) | 60% / 0.8 | 90% / 0.1 | 60% / 0.8 | 100% / 0.0 | 55% |
| thai_v3_m1024 (0.40 / 0.40) | 50% / 0.8 | 90% / 0.1 | 60% / 0.7 | 100% / 0.0 | 50% |
| thai_v4m_m1024 (0.50 / 0.40) | **20% / 2.4** | 45% / 1.1 | 25% / 1.6 | 100% / 0.0 | **10%** |

Per-photo total counts on s20 (true 82):
- thai_v2: 82 78 78 79 82 82 82 81 81 82 82 83 82 82 81 81 82 82 82 82
- thai_v3_m1024: 81 80 78 82 80 82 82 81 82 82 82 83 82 81 83 82 83 82 82 81
- thai_v4m_m1024: 82 75 75 87 84 83 81 81 83 82 86 85 83 82 83 86 82 81 84 88

thai_v4m_m1024 mostly **over-counts** s20 (12 of 20 photos above 82, up to +6), with two photos at −7.
The other two models stay within −4…+1. The per-form errors land on tablets and caplets, not mini tablets.

## 3. Speed (this Mac: Apple M1 Max, `device()` = CPU, imgsz 1024)
10 val photos (2 per session), 3 warm-up predictions, then 3 passes (30 timed predictions per model),
detector only (`predict`, including pre- and post-processing), compare thresholds.

| Model | median | mean | min–max |
|---|---|---|---|
| thai_v2 (11s) | 206 ms | 213 ms | 197–282 ms |
| thai_v3_m1024 (11m) | 394 ms | 398 ms | 373–452 ms |
| thai_v4m_m1024 (11m) | 396 ms | 402 ms | 376–468 ms |

thai_v4m_m1024 runs at the same speed as thai_v3_m1024, about 1.9× slower than thai_v2, as expected for the same architecture.

## 4. Reading
- **Validation:** adding the v4 step to the m model gave 82% exact / MAE 0.30. Exact is between thai_v2 (75%)
  and thai_v3_m1024 (85%), and MAE is the worst of the three. Most of thai_v3_m1024's lead came from the
  crowded tray s26, and v4m gives part of it back (93% → 80%, MAE 0.1 → 0.4).
- **Ordinary piles:** the picture is mixed. v4m is best on s13 (80% vs 60/60) and between the other two on s18. It is the only model that misses on s06
  (80%), and has the highest s23 MAE. Each of the small sessions has 5–10 photos, so one photo moves the rate by 10–20 points.
- **Mixed tray s20:** v4m is clearly worse than both references (20% total exact vs 60% / 50%; 10% all forms
  vs 55% / 50%), mostly from over-counting.
- **Speed:** unchanged from thai_v3_m1024 (~0.4 s/photo on CPU).
- In short, on these numbers the v4 step did not give the m model thai_v2's results on ordinary piles,
  and it did not keep thai_v3_m1024's crowded-tray (s26) result. The test set was not run.

## Files written
- `results/thresholds_thai_v2_compare.json`, `results/thresholds_thai_v3_m1024_compare.json` (new; its values match the
  app's `thresholds_thai_v3_m1024.json`), `results/thresholds_thai_v4m_m1024_compare.json`
- `results/thresholds_thai_v4m_m1024.json` (copy, for `eval_mixed.py`; not used by the app)
- This report. The speed script was a one-off and is not included.
