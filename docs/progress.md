# PillTally: Progress Report

> Working record from the project, lightly edited for this repository. Some paths it mentions are not
> included here: other model checkpoints and threshold files, `backups/`, `drive/`, `results/app_uploads/`
> (private photos), `results/s20_diag/`, `results/own_large_piles/`, `results/eval_other_pills_v2/` and the
> internal run notes. See the README for what is included.

Peeranat Patarakittisopol · as of 10 Oct 2026

## Summary

PillTally works end to end. A phone photo goes to a web app, which returns **the pill count with a box on every pill**, plus **each pill's form (tablet, caplet or mini tablet)** for checking that a tray holds only the expected form. A classical image-processing counter (no machine learning) is kept as the Level 2 evaluation baseline; it is no longer part of the app.

It's built from **two deep-learning models** that work one after the other:

```
Photo ─► ① COUNTER (thai_mix_s1024.pt)  1 class "pill"  → a box on every pill
             │ cut out each pill (masked crop, full resolution)
             ▼
         ② FORM CLASSIFIER (pill_type.pt, v3)   tablet / caplet / mini tablet
             ▼
         count + form of every pill; red box if a pill doesn't match the expected form
```

Both are trained with **supervised transfer learning**:
- **Counter (app):** `yolo11s.pt` (COCO) → `v4_pill_model.pt` (public pill photos) → **`thai_mix_s1024.pt`**: fine-tuned on our photos **plus public multi-drug photos** (Pill Counting v4 and the 110-drug Pill-Counter v2), so it counts other medicines too (sections 4.9–4.10). Earlier counters: `thai_v2.pt` (our photos only), `thai_v3_m1024.pt` (YOLO11m, our photos only).
- **Classifier:** `yolo11n-cls.pt` (ImageNet) → crops of our pills.

**Current results** (app counter `thai_mix_s1024`, conf 0.55 / IoU 0.6, chosen on validation; test set run twice, disclosed in 4.7 and 4.10):

| What | Result |
|---|---|
| Counter, validation (60 photos, 5 sessions incl. a crowded 90-pill tray; scored one photo at a time, as the app runs) | **93%** of photos counted exactly, average miss **0.10** pills (thai_v2: 77% / 0.30; thai_v3_m1024: 87% / 0.15) |
| Counter, **test set** (45 photos, 6 sessions; run 2, final model) | **89%** exact [78–98%], miss 0.13 (thai_v2 93%, thai_v3_m1024 82%; CIs overlap) |
| Counter, **other medicines** (187 held-out photos, Pill-Counter v2) | **58%** exact, miss 0.7 (thai_v2 23%, thai_v3_m1024 17%) |
| Classifier, validation (pills cut from the counter's own boxes) | **99.9%** of 784 pills |
| Classifier, held-out test crops (6 sessions) | **100%** of 1,110 pills |
| **Real mixed tray s20** (82 pills: 20 tablets + 40 caplets + 22 mini tablets, 20 photos) | total exact on **95%** of photos; all three forms exact on **90%** (thai_v2: 60% / 55%) |

**Main lessons learned:**
- The classifier took shortcuts three times, and each time a near-perfect score hid it. It used the **background**, then the **lighting**, then the **box size**. Each was caught with a targeted test and fixed with masking, new photos, and box-size augmentation (section 4.3).
- **Evaluated one way, deployed another:** for most of 10 Oct the threshold tuning scored validation photos in a different mode than the app (mixed photo sizes made the library pad every photo to a square). It picked IoU 0.4, which merged heaped caplets. Scoring photos one at a time, as the app does, fixed it (section 4.10).
- **A model trained only on our medicines doesn't generalize:** thai_v3_m1024 counted other medicines exactly on 11–17% of photos; adding public multi-drug photos to training raised that to 58% (sections 4.9–4.10).

**Main caveats:**
- The **test set was used twice** (run 1 at 14:40 on the earlier models; run 2 at 20:50 on the final model after new training data and the scoring fix). Both runs are reported. On the test set alone, thai_v2 is slightly ahead (93% vs 89%, overlapping CIs); the app counter was chosen on validation and external data, not on the test set.
- Mixed trays: s20 (floor tile, evaluation only) and four crowded trays on one beige tray. Unseen surfaces: the final counter missed 1–2 pills on 2 of 10 brown-cloth test photos.
- Tested up to **100 pills** (time test, 10 Oct) and 97 in the crowded mixed trays. Pharmacies mostly count 30, 60 or 90 tablets, so the claim is now **30–100 pills**; **150–300 is still untested**.

---

## 1. Data

### Open data (for the general pill model)
We reviewed four public datasets and used **Pill Counting v4** (Roboflow Universe, CC BY 4.0; must be credited): 1,236 photos of pharmacy counting trays, 71,269 labeled pills (1–202 per photo), one "pill" class. The others were rejected: too few pills per photo, blister packs, or built for identifying drugs.

### Our own photos: 159 photos in 22 sessions, plus 1 mixed tray

Each session is **one hand-counted pile of one form**, photographed 5–17 times.

| Form | Sessions | Photos | Pills | Surfaces and light |
|---|---|---|---|---|
| **Tablet** (round; white Cemol, and blue in s03/s04) | 14 | 80 | 3,230 | leather pad, brown cloth, granite, gray board, **floor tile**; daylight, warm indoor, indoor |
| **Caplet** (oblong) | 4 | 36 | 1,010 | floor tile, indoor |
| **Mini tablet** (small, yellow, round) | 4 | 43 | 722 | floor tile, indoor |
| **Mixed tray s20** (evaluation only) | 1 | 20 | 82 per photo (20 tablets + 40 caplets + 22 mini tablets) | floor tile, indoor |

- Pile sizes: 10–60 per single-form session, 82 in the mixed tray. One phone (iPhone).
- Session details: `data/sessions.csv`. Mixed-tray counts: `data/mixed_trays.csv`.
- **One photo was left out** (`s14_IMG_3292`): 2 mini tablets from another pile were at its edge, which would have taught the classifier that they're caplets. Kept in `data/real/excluded/` with a note.

### Labeling (one class, "pill")
| Round | Pre-label (by the current model) | After correction | How it was corrected |
|---|---|---|---|
| s01–s10 (tablets) | 1,937 boxes for 1,880 pills | **50 / 50 photos match** | Roboflow, then 4 speck boxes found by a size check and removed |
| s11–s20 (caplets, minis, mixed) | **68 / 80** single-form photos already exact | **79 / 80**, plus 1 left out | Roboflow (the remaining photo had real pills at its edge, see above) |
| s21–s24 (floor-tile tablets) | **28 / 30** exact | **30 / 30** | 2 false boxes on the floor pattern, found and removed by checking each |

**Final: 159 / 159 photos match their hand counts.** Pill forms are **not** labeled per box: they come from the session name, since each session holds only one form.

### Split by session (shared by both models, so neither trains on a session the other is tested on)
| Split | Sessions | Photos |
|---|---|---|
| Train | s01, s02, s05, s07, s09, s10 (tablets); s11, s12 (caplets); s15, s16 (minis); s21, s22 (floor-tile tablets) | 84 |
| Validation | s06 (tablet, granite), s13 (caplet), s18 (mini), s23 (tablet, floor tile) | 30 |
| Test | s03, s04 (**blue tablets, brown cloth, warm light: never seen in training**), s08 (tablet), s14 (caplet), s19 (mini), s24 (tablet, floor tile) | 45 |
| Evaluation only | s20 (mixed tray) | 20 |

**Honesty note:** during development a few test photos were looked at: an s03 photo (COCO classes), s04/s08 photos (checking the classifier server), and one s08 count was seen (47 vs 46). No setting was chosen from them, but the final report should mention it.

---

## 2. Models

### Level 4: general pill model (`v4_pill_model.pt`), the base for every counter
- Fine-tuned from `yolo11s.pt` (COCO) on Pill Counting v4 (1,054 train / 182 val) in Colab (T4): 1024 px, all layers trained, AdamW 0.002 (auto), batch 16, 60 epochs. The head was reduced from 80 classes to 1.
- v4 validation: **76.4%** exact after tuning (34.6% with the defaults; optimistic, since it was tuned on the same photos).
- On our first 50 tablet photos, with no training on them: 68% exact. Every error was on granite.

### Level 5: counters fine-tuned on our photos
| | `thai.pt` (first) | **`thai_v2.pt` (current)** |
|---|---|---|
| Trained on | 6 sessions, round tablets only (30 photos) | **12 sessions: tablets + caplets + mini tablets** (84 photos) |
| Where | Mac | **Colab, A100** (about 1.5–2 h on the Mac vs minutes on the A100) |
| Recipe | from v4 → stage 1: layers 0–9 frozen, SGD 0.005, 30 epochs → stage 2: all unfrozen, SGD 0.001, up to 60, early stop | same; batch 8 instead of 4 |
| Thresholds (tuned on validation) | conf 0.45, IoU 0.40 | conf 0.50, IoU 0.50 |
| **Validation: photos exact / average miss** | 77% / 0.27 | **90% / 0.10** |
| Per session: s06 tablet / s13 caplet / s18 mini / s23 floor tablet | 60% / 70% / 100% / 60% | **100% / 80% / 100% / 80%** |

`thai_v2` was chosen on **validation only**, then made the app's counter. `thai.pt` is kept. On 10 Oct `thai_v3_m1024` replaced `thai_v2` in the app (section 4.5).

### Form classifier (`pill_type.pt`, v3), "option B: detect, then classify"
- **Why a separate classifier instead of a 3-class counter:**
  - counting learns from every pill as one class
  - the classifier sees each pill at **full resolution** (stamps and edges visible)
  - pill forms come from session names, so no per-box form labels are needed
  - adding a form only means retraining the small classifier
- **Input:** a square crop around each pill box, from the full-resolution photo, with **everything outside an ellipse inside the box set to gray**, then 224 × 224.
- **Model:** YOLO11n-cls pretrained on **ImageNet** (1.28 M photos, 1,000 classes), fine-tuned on our crops. Rotation ±180°, flips, brightness ±40%, **little color change (saturation ±20%, hue 0)**, because color helps tell forms apart.
- **v3 training:** each training pill is cut 3 times: its box plus 2 copies **resized 0.85–1.2× and shifted up to ±10%**, so the result doesn't depend on exactly how the counter draws the box. 9,201 training crops. Trained on **Colab A100**.

### Level 2: classical image processing (`src/classical.py`, no machine learning)
1. Mark pixels whose color differs from the background (LAB distance, Otsu threshold), then remove specks.
2. Find pill centers with a distance transform.
3. Split touching pills with **watershed**.
4. Count.

Six settings were tuned on **train only** (about 2,400 combinations, search widened twice); best on train: MAE 4.9.

---

## 3. How the model works and was trained

### 3.1 Data pipeline (from taking the photo to the model input)

| # | Step | What was done | Why |
|---|---|---|---|
| 1 | Take photos | iPhone, 4032×3024 HEIC; one hand-counted pile of one form per session; folders named `session-form-count` | Sessions allow a leakage-free split; the folder name gives the form and the count |
| 2 | HEIC → JPG | macOS `sips`, quality 92; originals kept in `data/real/originals/` | Most ML tools can't read HEIC |
| 3 | Fix rotation | EXIF rotation applied to the pixels, then removed | Tools disagree on the tag, which would shift boxes |
| 4 | Metadata | `data/sessions.csv` (one row per photo); `data/mixed_trays.csv` | Needed by the split, the evaluation and the label check |
| 5 | Pre-label | The current counter writes YOLO labels: `class x_center y_center width height`, as fractions of the image | Correcting is much faster than drawing |
| 6 | Correct in Roboflow | Fixed by hand, exported as YOLOv11. Roboflow's 512×512 copies are ignored; the labels (fractions) fit the originals | Accurate labels |
| 7 | Check labels | Box count vs hand count for every photo, plus visual checks of suspect boxes | Wrong labels teach and evaluate wrongly |
| 8 | Split by session | `split.py` (counter) and `crops.py` (classifier) use the same held-out sessions | Lecture 3's same-session leakage pitfall |
| 9 | Crops (classifier) | `crops.py`: masked square crops from full resolution, labeled by session; 2 extra box-varied copies per training pill | See 3.5 |

**Public data (v4):** shrunk once to 1280 px for Colab speed (labels unchanged). **Colab uploads** of our photos are shrunk to 2048 px for the same reason.

**Inside training (done automatically by Ultralytics):**

| Step | What happens | Train | Val / test / app |
|---|---|---|---|
| Letterbox resize | Fit into 1024×1024 (counter) or 224×224 (classifier), proportions kept | ✅ | ✅ |
| Pixel scaling | 0–255 → 0–1 | ✅ | ✅ |
| Mosaic (counter) | 4 photos tiled into one (off for the last 10 epochs) | ✅ | ❌ |
| Geometric augmentation | Rotation ±180°, flips, scale ±30%, shift ±10% | ✅ | ❌ |
| Photometric augmentation | Counter: brightness ±40%, saturation ±50%. Classifier: brightness ±40%, saturation ±20% | ✅ | ❌ |
| Rare extras (Albumentations) | Blur, median blur, grayscale, CLAHE, each at 1% | ✅ | ❌ |

**In the app:**
1. The browser applies the photo's rotation and sends it at **full resolution** (up to 4032 px, about 3–4 MB), because the classifier crops each pill from it.
2. The server applies the rotation again and converts RGB → BGR.
3. The counter letterboxes to 1024; the classifier gets masked 224 px crops. This is the same preprocessing as in validation and test.

**Color, not grayscale.** Both models use RGB. Color separates pills from busy backgrounds, and it helps tell forms apart (the mini tablets are yellow). Over-reliance on color is limited by brightness/saturation augmentation and 1% grayscale. Grayscale is the better choice when color carries no information (X-rays, documents, cracks in metal).

### 3.2 What a CNN is, and the YOLO11s architecture

- **CNN (convolutional neural network):** small learned filters (e.g. 3×3 = 9 weights) slide over the image. Each makes a feature map that is high where its pattern appears. Stacked layers go from edges → shapes → objects; stride 2 or max pooling shrinks the map. Filters are learned by backpropagation, not written by hand. Lecture 3's properties: local connectivity, weight sharing, translation invariance.
- **Why not fully connected:** one dense layer of 1,000 neurons on a 224×224 color image needs 150 M weights (3.1 B at 1024×1024). All of YOLO11s has **9,428,179** weights (1-class head).
- **Real feature maps** from the counter on our photo s01: layer 1 reacts to brightness and stamp texture, layer 4 to curved outlines, layer 6 gives one spot per pill. (Channels were chosen by best match to edges, pill discs or centers, to illustrate.)
- **24 top-level layers** (181 small operations):

| Layers | Part | Job | Map size at 1024 input |
|---|---|---|---|
| 0–8 | Backbone | Stride-2 convolutions alternating with C3k2 residual blocks | 512 → 32 |
| 9 | Backbone | SPPF: repeated 5×5 max pooling, wider context | 32 |
| 10 | Backbone | C2PSA: small attention block | 32 |
| 11–22 | Neck | Upsample + concatenate + C3k2 (FPN/PAN): mixes detail with context | 128, 64, 32 |
| 23 | Head (Detect) | Anchor-free: box + pill score at every cell of 3 grids | 21,504 candidate boxes |

- Pills are about **54 px** at 1024 input (about 7 cells of the finest grid), against 34 px at the default 640. That's why 1024 is used.
- **COCO** (about 330 k everyday photos, 80 classes) is the counter's base. It has no "pill" class: on our photos it saw "39 frisbees" on one, 1 frisbee on another.
- **ImageNet** (about 1.28 M photos, 1,000 classes, one label per image) is the classifier's base.

### 3.3 How the counter learns
1. **Matching:** the Task-Aligned Assigner picks, for each labeled pill, the 10 grid cells with the best mix of score and overlap; all other cells learn "no pill".
2. **Loss:** 7.5 × CIoU (box overlap, center, shape) + 1.5 × DFL (each box side as a 16-bin distribution) + 0.5 × BCE (pill or not).
3. **Update:** backpropagation + gradient descent; the best epoch on validation is kept; early stop after 25 epochs without improvement.

### 3.4 Fine-tuning method: gradual unfreezing (counter)

| | v4 step | Thai stage 1 | Thai stage 2 |
|---|---|---|---|
| Lecture 3 approach | Full fine-tuning (plenty of data) | **Fine-tune top layers** (backbone frozen) | **Full fine-tuning**, low learning rate |
| Frozen | nothing | layers 0–9: **4.45 M weights (47%)** | nothing |
| Optimizer, learning rate | AdamW 0.002 (auto) | SGD 0.005 | SGD 0.001 (5× lower) |
| Warm-up | 3 epochs | 3 epochs | none |

Learning rates follow Ultralytics' default **linear decay** after warm-up.

**Why freeze first (less data → freeze more):**
- protects the general features learned from COCO and v4
- reduces overfitting
- keeps the start of training stable
- trains faster

**Honest limits:** the benefit of freezing wasn't measured (no run without it), and the learning rates are standard choices, not tuned.

### 3.5 How the form classifier avoids shortcuts
| Shortcut found | How it was found | Fix |
|---|---|---|
| **Background and neighbors:** a blue pill among white ones was called white 89% of the time, while normal test accuracy was 100% | Pasting a pill onto the other class's background | **Mask** everything outside the pill (ellipse inside the box) |
| **Lighting:** all caplets and minis were photographed on the floor tile, and no tablets were, so "floor-tile light + white = caplet". On the real mixed tray, **tablets: 0 / 20 photos right** | The **real mixed tray**; the pasted test missed it, because a pasted pill keeps its own photo's light | **New photos:** tablets on the floor tile (s21–s24) |
| **Box size:** with the new counter's boxes, tablets dropped to **84%** on validation | Validation using the **counter's own boxes** instead of hand-checked ones | Train on **box-varied crops** (v3) → **99.7%** |

**Remaining known limit:** each crop is resized to 224 px, so the classifier can't use a pill's **real size**, only color and shape.

---

## 4. Results (validation and the mixed tray; the test set is still unused)

### 4.1 First baseline comparison (session s06 only, 39 tablets, `thai.pt`)
| Level | Method | Counts (5 photos) | Exact | MAE |
|---|---|---|---|---|
| 1 | Predict the average train count | 38 ×5 | 0% | 1.00 |
| 2 | Classical image processing (no ML) | 40, 23, 21, 12, 19 | 0% | 16.40 |
| 3 | Pretrained YOLO, no pill training | 39, 35, 37, 34, 36 | 20% | 2.80 |
| 4 | v4 pill model | 38, 39, 41, 40, 38 | 20% | 1.00 |
| 5 | v4 + fine-tune | 39 ×5 | 100% | 0.00 |

**Why Level 2 fails:** shadows join pills into long blobs, touching pills merge, and pills smaller than its tuned size (a photo taken from farther away) are ignored. **Level 1 looks good** only because all piles were 25–50 pills.

### 4.2 Counter, current validation (s06, s13, s18, s23)
See the table in section 2: `thai_v2` **90% exact / 0.10 miss** vs `thai.pt` 77% / 0.27.

### 4.3 Form classifier
| | Result |
|---|---|
| Held-out test crops (hand-checked boxes, 6 sessions incl. blue tablets never seen in training) | **1,110 / 1,110 (100%)** |
| Validation, pills cut from **the counter's boxes** | **99.9%** (v2 was 92.1%) |
| **One wrong pill hidden in a tray** (pasted, 600 cases, held-out photos) | caught **100%** in all 6 combinations; correct pills wrongly flagged: 0–2% |

### 4.4 Real mixed tray s20: how the full system improved
| Counter + classifier | Total exact | Tablets (20) | Caplets (40) | Minis (22) | **All 3 exact** |
|---|---|---|---|---|---|
| thai + classifier v1 (before floor-tile tablets) | 35% | **0%** | 0% | 60% | **0%** |
| thai + classifier v2 (floor-tile tablets added) | 40% | 90% | 50% | 90% | 40% |
| thai_v2 + classifier v2 | 60% | 50% | 35% | 80% | 30% |
| **thai_v2 + classifier v3 (current)** | **60%** | **90%** | **60%** | **100%** | **55%** |

- Row 1 uses `thai.pt` with its first thresholds (conf 0.35, tuned on s06 only). Row 2 uses `thai.pt` re-tuned on the 4 validation sessions (conf 0.45); with the first thresholds, row 2's classifier scored 35% / 85% / 35% / 90% / 30%.
### 4.5 Experiment: training on crowded mixed trays (10 Oct)
Four new crowded mixed trays on a **beige plastic tray** (s25–s28, 90–97 pills each, 97 photos, recounted by hand; 117 / 117 mixed photos match). s25, s27, s28 were added to training and s26 to validation; s20 stayed untouched. Three counters were trained on the Colab A100 and tuned the same way on validation (`tools/compare_counters.py`):

| Counter | Validation exact / miss (60 photos) | Crowded tray s26 | Mixed tray s20: total exact / miss / all 3 forms | Speed |
|---|---|---|---|---|
| thai_v2 (app) | 75% / 0.28 (re-tuned) | 67% | **60% / 0.8 / 55%** (original thresholds) | 0.16 s |
| thai_v3_s1024 | 83% / 0.20 | 80% | 30% / 2.0 / 25% | 0.16 s |
| thai_v3_s1280 | 78% / 0.30 | 80% | — | — |
| thai_v3_m1024 (YOLO11m, from COCO, no v4 step) | **85% / 0.20** | **93%** | 50% / 0.8 / 50% | 0.33 s |

- The crowded mixed trays **clearly helped on validation**, especially on the crowded tray s26 (67% → 93%).
- **1280 px didn't help** (worse than 1024).
- **On the floor-tile mixed tray s20 there was no gain** (same average miss, 1–2 fewer exact photos). A likely reason: all new trays are on the beige tray, s20 is on the floor tile, so the new examples don't carry over to another surface.
- Crowded counts are **sensitive to thresholds**: thai_v2 scored 10% on s20 with thresholds re-tuned on the larger validation set, against 60% with its original ones.
- **First decision: the app kept thai_v2**, because thai_v3 didn't gain on s20.
- **Changed on 10 Oct: the app now uses thai_v3_m1024** (conf 0.40, IoU 0.40, `results/thresholds_thai_v3_m1024.json`). It's the best on validation (85% vs 75%) and on the crowded tray s26 (93% vs 67%). The trade-offs: on the floor-tile mixed tray s20 it gets 1–2 fewer exact photos (same average miss), and the counter takes about twice as long (0.33 s vs 0.16 s). thai_v2 stays in `results/models/`; to go back, run the app with `PILLTALLY_MODEL=results/models/thai_v2.pt PILLTALLY_THRESHOLDS=thresholds_thai_v2.json`.
- Next step for this: a crowded mixed tray on the floor tile (and other surfaces) in training.

- **Where the errors are now: counting in the crowded middle.** When the counter adds or misses a box, it usually shows up in the caplet total. Diagonal caplets, whose boxes are square, are classified correctly.
- Scores are from **totals per photo**. A true per-pill score would need form labels on each box of a few mixed photos.

---

### 4.6 Time test and cost (10 Oct)
Same piles counted by hand (five at a time) and with the app (`thai_v3_m1024`), one try each; results in `data/time_test.csv`.

| Pile | By hand | PillTally (processing) | App count |
|---|---|---|---|
| 10 | 3.55 s | 0.4 s | 10 ✅ |
| 50 | 22 s | 0.5 s | 50 ✅ |
| 100 | 48.81 s | 0.6 s | 100 ✅ |
| **Per pill (total time / 160 pills)** | **0.46 s** | **0.009 s** | about **50× faster** |

- The PillTally time is the processing time after the photo; taking the photo adds a few seconds. Even with 10 s added, 100 pills take about a fifth of the hand time, past the proposal's target of 50% less time.
- Hand time per pill grows with the pile (0.36 → 0.49 s); the app's time per photo barely changes.
- Very close-up photos (taken before this test) missed a caplet on a 10-pill tray; photos from normal height were exact.

**Cost.** Counting machines: GSE Eyecon from about $7,500, Kirby Lester KL1 from about $4,700, RxCount Rx-4 $2,499 (RxCount 2019 list prices); counters sold in Thailand ฿49,000–125,900 (mostly bottling counters). PillTally: staff phones plus a server on an existing PC (no extra hardware), or a small cloud server (AWS c7g.large in Thailand, $0.0708/h, about $15 a month for 8 h × 26 days). The machines are certified and need no network; PillTally is an assistant, and its speed on a small cloud CPU isn't measured yet.

### 4.7 Final test set, run once (10 Oct)
`python tools/evaluate.py report --split test --final` (logged in `results/test_log.txt`; full table in `results/test_results.md`). 45 photos, 6 single-form sessions of 10–46 pills (s03, s04 unseen blue tablets on brown cloth). Every model and threshold was chosen before this run.

| Level | Exact [95% CI] | MAE [95% CI] |
|---|---|---|
| L1 mean train count | 0% | 37.33 |
| L2 classical watershed | 4% [0, 8] | 9.09 |
| L3 pretrained YOLO (COCO) | 0% | 23.18 |
| L4 v4 pill model | 82% [57, 95] | 0.69 |
| L5 thai_v2 (YOLO11s, via v4) | **93%** [91, 97] | **0.07** |
| L5 thai_v3_m1024 (app counter) | 84% [76, 97] | 0.18 |

- Every learned counter beats the baselines by a wide margin; on the unseen condition (s03, s04) L4 and both L5 models are 10 / 10 exact.
- **The app's counter (m1024) is behind thai_v2 on this test set** (84% vs 93%). Its errors are on the small caplet and mini-tablet piles (s14: 15 caplets, s19: 10 minis), off by 1–2. The CIs overlap.
- With the mixed tray s20 (v2 60% vs m1024 50%), both held-out checks favor thai_v2 on single-form and floor-tile piles; m1024 was chosen on validation for crowded trays (s26: 93% vs 67%). Switching now would be a choice made on the test set, so it must be reported as such.
- L1 is worse than before (mean train count now 62) because the crowded trays raised the training average.

### 4.8 Experiment: the m model through the v4 public data (10 Oct, after the test run)
Question from 4.7: did `thai_v2` win the test set because of the v4 public-pill step? `thai_v4m_m1024` = YOLO11m, COCO → v4 → our photos, same recipe as v3 (`notebooks/train_counter_v4m_colab.ipynb`). Compared on validation and s20 only; the test set was not used again. Full report: `docs/eval_v4m_report.md`.

| | thai_v2 (s, via v4) | **thai_v3_m1024 (m, app)** | thai_v4m_m1024 (m, via v4) |
|---|---|---|---|
| Validation exact / miss (60 photos) | 75% / 0.28 | **85% / 0.20** | 82% / 0.30 |
| Crowded tray s26 | 67% | **93%** | 80% |
| Caplets s13 | 60% | 60% | 80% |
| Mixed tray s20: total exact / all 3 forms | **60% / 55%** | 50% / 50% | 20% / 10% (over-counts, up to +6) |
| Detector time per photo (Mac CPU) | **0.21 s** | 0.39 s | 0.40 s |

- **The v4 step did not help the m model.** It gave back most of the crowded-tray gain and over-counted the mixed tray. This partly answers the "no v4 step" ablation: for the m model, adding the public data did not help on our data.
- **Decision: the app keeps `thai_v3_m1024`**, chosen on validation (best overall and on crowded trays). The trade-off stays as reported in 4.7: thai_v2 is better on small single-form piles, on s20 and in speed.

### 4.9 Other medicines: does the counter generalize? (10 Oct)
Counted public photos of medicines we never photographed, with each model's app thresholds and no re-tuning
(`notebooks/eval_other_pills_colab.ipynb`, run on Colab; report in Drive `MyDrive/PillTally/eval_other_pills/`).
Every label counts as one pill.

| Dataset | thai_v3_m1024 | thai_v2 |
|---|---|---|
| Pill-Counter v2: 1,255 photos, 110 drugs, piles of 1–18 | 11% exact, MAE 3.7 | **36%**, MAE 1.6 |
| ↳ photos mixing several named drugs, one pill each (268) | 5%, MAE 4.4 (misses pills) | **83%**, MAE 0.2 |
| ↳ brown capsule / white tablet / yellow tablet groups (17–18 photos each) | 44% / 41% / 71% | 39% / 47% / 59% |
| Pill Counting v4 validation: 182 tray photos, 4–180 pills | 32%, over-counts (bias +9.0) | 66% (seen-ish: v4-stage validation set) |
| pill count v1 tray video frames: 112 frames, 640×480, 10–175 pills | 1%, under-counts | 0%, under-counts |

- **thai_v3_m1024 learned our three medicines, not "pills in general".** On photos of several different drugs it misses
  pills almost every time. thai_v2, which went through 71k public pills first, generalizes much better, though still far
  from good on unseen drugs (36% exact on Pill-Counter v2).
- The v1 frames are low-resolution video stills; both models under-count there, so that row says more about the data
  than about the models.

**Decision (10 Oct): the app switches back to `thai_v2`** (conf 0.50, IoU 0.50). Reason: generalization to other
medicines, measured on external data, not on the test set. For the record, the test set (4.7), the mixed tray s20 and
speed also favor thai_v2; thai_v3_m1024 stays better only on our crowded beige trays (s26).
**Next:** a counter trained on our photos **plus** public multi-drug photos, chosen on our validation **and** a held-out
"other medicines" check.

### 4.10 Scoring fix, mixed-data counter, final test run (10 Oct, evening)
Full report: `docs/eval_mix_report.md`.

**The scoring bug.** `compare_counters.py` passed all validation photos to the detector as one list. Eight s26 photos are
5712×4284 (the rest 4032×3024), and with mixed sizes Ultralytics pads every photo to a 1024×1024 square, while the app
predicts one photo at a time (1024×768). So every validation number and threshold picked with that script after s26 joined
validation (sections 4.5, 4.8, 4.9 and the "85%" for thai_v3_m1024) was measured in a mode the app doesn't use. Not affected:
the test runs (all test photos are one size), thai_v2's thresholds (tuned before s26), s20, the Colab runs and speed.
Fixed: `compare_counters.py` and `evaluate.py` now predict one photo per call (old outputs in `backups/eval_fix_2026-10-10/`).

**Mixed-data counter `thai_mix_s1024`:** v4 pill model → our photos (×6, about a third of each epoch) + Pill Counting v4 +
Pill-Counter v2 (110 drugs, re-split by prescription / shoot so near-duplicates never cross parts; 15% held out as an
"other medicines" test). Notebook `notebooks/train_counter_mix_colab.ipynb`, Colab A100.

| (all scored one photo at a time; thresholds tuned on validation) | thai_v2 | thai_v3_m1024 | thai_v4m_m1024 | **thai_mix_s1024 (app)** |
|---|---|---|---|---|
| Thresholds conf / IoU | 0.50 / 0.50 (app file) | 0.25 / 0.5 | 0.60 / 0.5 | **0.55 / 0.6** |
| Validation exact / miss, photo-weighted (60 photos) | 68% / 0.37 (77% / 0.30 at 0.55 / 0.6) | 87% / 0.15 | 82% / 0.25 | **93% / 0.10** |
| Crowded tray s26 | 63% | 87% | 80% | **93%** |
| Mixed tray s20: total / all 3 forms | 60% / 55% | 75% / 75% | 55% / 50% | **95% / 90%** |
| Other medicines, held-out 187 photos: exact / miss | 25% / 1.9 | 17% / 3.6 | 37% / 1.6 | **58% / 0.7** |
| Own large piles (10, 50, 60, 100, 100, 100) | all within ±1 | within ±1 | within ±1 | 10, 50, 60, 100, 99, 100 |
| Test set run 2: exact [95% CI] / miss | **93%** [91–97] / 0.07 | 82% [71–93] / 0.20 | – | 89% [78–98] / 0.13 |
| Detector time per photo (Mac CPU) | 0.19 s | 0.38 s | 0.38 s | 0.20 s |

- **Why the s20 caplets were missed before:** at IoU 0.4, 44 of 49 missed caplets had been detected and then removed by the
  overlap filter (heaped caplets). At the IoU 0.6 chosen by correct validation scoring, s20 is 95% exact.
- **Decision: the app uses `thai_mix_s1024` at conf 0.55 / IoU 0.6**, chosen on validation (best overall, best on crowded
  trays and the mixed tray) and the external "other medicines" check. On the test set it is close to thai_v2 (89% vs 93%,
  overlapping CIs); its 5 misses are −1/−2 on unseen brown cloth (2 photos) and caplets (3 photos).
- **Test set, run 2 (disclosed):** run at 20:50 after the model and thresholds were fixed, logged in `results/test_log.txt`;
  run 1 is kept in `backups/test_run1_2026-10-10/`.
- Still weak for every model: low-resolution video frames of 100–175 pills (pill count v1), where all counters under-count.

## 5. Demo app

```
Phone browser ─► Next.js page (port 3000) ─► FastAPI model server (port 8000)
  full-resolution photo   forwards /api/*          thai_mix_s1024.pt counter + pill_type.pt classifier
```

- **Start:** `./dev.sh` (one command; prints the phone address; Ctrl+C stops both).
  - The server **reloads by itself** when `src/` code or a model in `results/models/` changes.
  - The page updates itself when `web/` changes, and re-checks the form list every 15 s.
- **Marks:** green boxes are confident, orange ones uncertain. An optional hand-count field shows the count's error.
  (An earlier version also had an "Old way" tab with the classical counter; it was removed from the app, and the
  classical method was used only as the report's Level 2 baseline, whose code is not in this repository.)
- **Pill form (experimental):** pick tablet / caplet / mini tablet. Every pill gets a form, and pills of another form, or whose form is unsure (below 80%), get **red boxes**.
- **Upload log:** every photo and its result are saved locally in `results/app_uploads/`. They're used to find failure cases and turn them into new training data.
- **Speed:** about 0.3–0.7 s on the server per photo (counter + classifier), plus upload, measured with thai_v2. The thai_v3_m1024 counter adds about 0.17 s per photo.

---

## 6. Compared with the proposal and the course

| Proposal item | Status |
|---|---|
| Own dataset, split by session | ✅ 159 photos, 22 sessions + 1 mixed tray |
| Labels checked against hand counts | ✅ 159 / 159 |
| Transfer learning, freeze then unfreeze, 1024 px | ✅ |
| Thresholds tuned on validation | ✅ |
| Baselines, Levels 1–4 | ✅ (first validation); to be repeated on the final test |
| Unseen condition in test | ✅ blue tablets, brown cloth, warm light (s03, s04) |
| More pill types | ✅ 3 forms (tablet, caplet, mini tablet), plus a form classifier (beyond the proposal) |
| Piles of 100–300, touching | ⚠️ rescoped to 30–100: tested to 100 (exact); 150–300 untested |
| Test set run once, with confidence intervals | ⏳ |
| Ablations (no v4 step, no augmentation, no freezing) | ⚠️ partly: v4 step for the m model tested (no gain, section 4.8); the others not run |
| Time test against hand counting | ✅ 0.46 s vs 0.009 s per pill (section 4.6) |
| Demo on a phone | ✅ |
| Demo deck | ✅ `docs/slides.pptx` (33 slides) updated 10 Oct evening: app counter `thai_mix_s1024`, corrected validation, other medicines, both test runs disclosed |

**Course alignment** (course lecture slides, not included):
- **Lecture 3:** CNN properties, stride, max pooling, residual blocks, object detection, transfer learning (less data → freeze more), augmentation, the same-session leakage pitfall, distribution shift (the lighting shortcut).
- **Lecture 5:** product metric (exact count) + MAE; train/val/test with test once; baseline types; "be suspicious of near-perfect results" (three times in this project); ablations; confidence intervals.

---

## 7. Next steps

| # | Task |
|---|---|
| 1 | **Test the app on real trays** (especially tablets and caplets mixed); failures are saved in `results/app_uploads/` |
| 2 | **Turn failures into training data:** correct their boxes, add them as sessions, retrain (an active-learning loop) |
| 3 | **More variety:** caplets and minis on other surfaces and in other light (both are only on the floor tile now) |
| 4 | **A new mixed tray with per-box form labels** (3 classes in Roboflow) for training, and a per-pill score on s20 |
| 5 | **Larger piles (150–300 touching)**; more medicines and shapes (only the classifier retrains); real pill size from a reference card; barcode scan for identical drugs; run on the phone; a pharmacy pilot |
| 6 | **Run the test set once:** `python tools/evaluate.py report --split test --final`, plus `tools/eval_mixed.py` |
| 7 | Ablations, time test |
| 8 | Update the deck (forms, classifier, `thai_v2`, the shortcut story); README for the code |

---

## 8. Files

| Path | What |
|---|---|
| `dev.sh` | Start the app (auto-reloading) |
| `data/sessions.csv`, `data/mixed_trays.csv`, `data/time_test.csv` | Hand counts, session details, time test |
| `data/real/images/`, `labels/`, `excluded/` | Photos, checked labels, left-out photo (the original HEIC files are not included) |
| `data/real/{train,val,test}.txt`, `split.csv` | Counter split |
| `data/crops/` | Classifier crops (not included; rebuilt by `src/crops.py`) |
| `notebooks/` | Colab training and evaluation notebooks (v4 pill model, counters, form classifier, other medicines) |
| `tools/prelabel.py`, `split.py`, `finetune.py` | Pre-label, split, two-stage fine-tuning |
| `src/crops.py`, `train_classifier.py`, `test_odd_pill.py` | Classifier crops, training, wrong-pill test |
| `tools/evaluate.py`, `eval_mixed.py`, `compare_counters.py` | Thresholds + Levels 1–5 report; mixed-tray score; counter comparison |
| `src/classical.py` | Level 2 classical counter |
| `src/api.py`, `web/` | Model server and phone page |
| `results/models/` | `thai_mix_s1024.pt` (app), `pill_type.pt` (= v3, app), `thai_v2.pt`, `thai_v3_m1024.pt`, `v4_pill_model.pt` (the other checkpoints are not included) |
| `results/thresholds_*.json` | Tuned thresholds (`thresholds_thai_mix_s1024_single.json` used by the app; `*_single` / `*_compare` = scored one photo at a time) |
| `docs/slides.pptx`, `docs/proposal.md`, `docs/PillTally_report.docx` | Deck, proposal, report |
