# PillTally

PillTally counts pills from one phone photo, to assist the pharmacist's check when filling a prescription.
The pharmacist spreads the pills on a tray and takes a photo; a detector draws a box on every pill (count = boxes)
and a small classifier names each pill's form (tablet, caplet, mini tablet), so a pill of the wrong form gets a red
box. Uncertain boxes are shown in orange for a second look. It is an aid, not a certified device: the pharmacist
sees every box and stays responsible for the count.

Applied AI course project · Peeranat Patarakittisopol · October 2026 · Full report: [`docs/PillTally_report.docx`](docs/PillTally_report.docx)

```
Phone browser ─► Next.js page (web/, port 3000) ─► FastAPI model server (src/api.py, port 8000)
                                                     ① counter   results/models/thai_mix_s1024.pt
                                                     ② forms     results/models/pill_type.pt
```

## Results

| Check | Result |
|---|---|
| Counter, **test set** (45 photos, 6 held-out sessions) | **89%** of photos counted exactly (MAE 0.13 pills) |
| Counter, validation (60 photos, 5 sessions incl. a crowded 90-pill tray) | **93%** exact (MAE 0.10) |
| Real mixed tray (82 pills, 3 forms) | total exact on **95%** of photos; all three form counts exact on **90%** |
| Other medicines (187 held-out public photos of drugs we never photographed) | **58%** exact (MAE 0.7) |
| Form classifier, per pill | **99.9%** on validation, **100%** on 1,110 held-out test pills |
| Time (10, 50, 100 pills) | **0.009 s** per pill (processing) vs **0.46 s** per pill by hand; all counted exactly |

Baselines on the same test set: COCO-pretrained YOLO 0% exact, public pill model 82%.
The report's Level 2 baseline (classical image processing: 4% exact on the test set) was computed with code not included in this repository.
**Disclosure:** the test set was run twice (both runs are logged in `results/test_log.txt`); the 89% is from the
second run, with the final model fixed beforehand. The time test was measured with the earlier `thai_v3_m1024` counter.
Details and caveats: [`docs/progress.md`](docs/progress.md) (sections 4.5–4.10) and [`docs/eval_mix_report.md`](docs/eval_mix_report.md).

## Quick start

Needs macOS or Linux, Python 3.11+ (tested on 3.13) and Node 20+. The trained models are included in `results/models/`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd web && npm ci)
```

Start the two servers, each in its own terminal (Ctrl+C stops each one):

```bash
.venv/bin/uvicorn api:app --app-dir src --port 8000 --reload --reload-dir src   # model server
cd web && npx next dev -H 0.0.0.0 -p 3000                                        # phone page
```

Open `http://<your-Mac-IP>:3000` on a phone on the same Wi-Fi (`ipconfig getifaddr en0` prints the IP),
and photograph pills spread in one layer on a plain tray, with the whole tray in view.
Each uploaded photo and its result are saved to `results/app_uploads/` (not tracked by git; `PILLTALLY_LOG_UPLOADS=0` turns this off).
To try another counter: set `PILLTALLY_MODEL=results/models/thai_v2.pt PILLTALLY_THRESHOLDS=thresholds_thai_v2.json` before the uvicorn command.

### With Docker

Needs only Docker (Docker Desktop on a Mac). The models stay in `results/`, which is mounted into the container.

```bash
docker compose up --build        # first build downloads ~1-2 GB; later starts are quick
```

Then open `http://<your-Mac-IP>:3000` on the phone (`ipconfig getifaddr en0` prints the IP). Docker runs the models on the
CPU, so a count takes longer than with the two commands above, which use the Mac's GPU. Both use ports 3000 and 8000: run one at a time.
`docker compose down` stops it.

## Repository layout

```
PillTally/
├── requirements.txt   Python packages (pinned)
├── src/               Python backend: model server (api.py), shared paths (common.py), pill crops (crops.py)
├── tools/             Python scripts for data prep, training and evaluation (see tools/README.md)
├── notebooks/         Google Colab notebooks for GPU training and the other-medicines check
├── web/               Next.js phone page
├── data/              our photos, labels, hand counts and splits (public datasets not included)
├── results/           trained models, tuned thresholds, test and validation reports
└── docs/              report, proposal, slides, progress record, evaluation reports
```

## Reproduce

Data flows top to bottom. Steps marked *Colab* ran on a Google Colab GPU; they read zips from Google Drive
(`MyDrive/PillTally/`), and each notebook lists the files it needs at the top.

| Step | Code | What it does |
|---|---|---|
| Public pill model | Colab notebook, archived (not in this folder) | YOLO11s (COCO) fine-tuned on *Pill Counting v4* → `v4_pill_model.pt` |
| Pre-label our photos | `tools/prelabel.py` | The model writes YOLO labels; the boxes were then corrected in Roboflow and checked against hand counts |
| Split by session | `tools/split.py` | Photos of one pile never sit in two splits; tags unseen conditions |
| Counter on our photos | `tools/finetune.py`; `notebooks/train_counter_mix_colab.ipynb` (Colab, app counter) | Two stages: backbone frozen (SGD 0.005), then all layers (SGD 0.001), early stop |
| App counter | `tools/make_mixed_zip.py` → `notebooks/train_counter_mix_colab.ipynb` (Colab) | Our photos + Pill Counting v4 + Pill-Counter v2 (re-split by source group) → `thai_mix_s1024.pt` |
| Compare counters | `tools/compare_counters.py` | Tunes conf / NMS IoU per model on validation only, one photo per call like the app |
| Form classifier | `src/crops.py`, `tools/train_classifier.py` or `notebooks/train_classifier_colab.ipynb` (Colab) | Masked full-resolution crops of each pill, labeled by session; YOLO11n-cls (ImageNet) → `pill_type.pt` |
| Wrong-pill test | `tools/test_odd_pill.py` | Pastes one pill of another form into a tray: does the classifier catch it? |
| Evaluation | `tools/evaluate.py`, `tools/eval_mixed.py` | Levels 1 and 3–5 with session-bootstrap CIs; the test set is guarded in code (`--final`, logged in `results/test_log.txt`); mixed-tray score per form |
| Other medicines | `tools/make_otherpills_zip.py` → Colab notebook, archived (not in this folder) | Counts public photos of medicines we never photographed → `results/eval_other_pills_v3/` |
| App | `src/api.py`, `web/app/page.tsx`, `docker-compose.yml` | Model server and phone page |

`notebooks/train_counter_mix_colab.ipynb` is written by `notebooks/make_train_counter_mix_nb.py`;
edit that and re-run it. The zips of our own data for Colab are plain archives of the data folders:
`cd data && zip -r ../pilltally_real_v3_for_colab.zip real/images real/labels real/*.txt` and, after `src/crops.py`,
`cd data && zip -r ../pilltally_crops_for_colab.zip crops`.

Validation can be re-run at any time: `.venv/bin/python tools/evaluate.py report --split val`.

## Data and licenses

- **Our photos (included):** `data/real/` holds the 276 photos taken for this project (27 sessions: 256 photos listed in
  `data/sessions.csv` plus 20 of the mixed tray s20, which is evaluation only), their checked YOLO labels (every photo's
  box count equals its hand count), the split files and one excluded photo. Per-form counts of the mixed trays are in
  `data/mixed_trays.csv` and the timings in `data/time_test.csv`.
- **Public datasets (not included; download in YOLOv11 format from Roboflow Universe and unzip into `data/open/`):**

  | Dataset | License | URL | Folder |
  |---|---|---|---|
  | Pill Counting v4 | CC BY 4.0 | https://universe.roboflow.com/jeonghyeon-kim/pill-counting-im4ph/dataset/4 | `data/open/Pill Counting.v4i.yolov11/` |
  | Pill-Counter v2 | Public Domain | https://universe.roboflow.com/dubfai/pill-counter-guifm/dataset/2 | `data/open/Pill-Counter.v2i.yolov11/` |
  | pill count v1 | CC BY 4.0 | https://universe.roboflow.com/ncsang1/pill-count-mfuny/dataset/1 | `data/open/pill count.v1i.yolov11/` |

  Our re-split of Pill-Counter v2 by source group is included as `data/open/pillcounter_v2_split.csv`.

## Limits

Tested up to 100 pills (150–300 not yet); crowded trays only on one surface; the classifier can't see real pill size, so
identical-looking drugs aren't told apart (the drug is identified from the stock bottle; the tool counts).

## Credits

Built with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0), PyTorch, OpenCV,
FastAPI and Next.js. Public training and evaluation data from the Roboflow Universe datasets above, used under their licenses.
Report: [`docs/PillTally_report.docx`](docs/PillTally_report.docx) · proposal: [`docs/proposal.md`](docs/proposal.md) ·
slides: [`docs/slides.pptx`](docs/slides.pptx).
# PillTally
