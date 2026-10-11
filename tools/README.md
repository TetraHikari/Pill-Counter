# tools

Scripts for building the data, training the models and evaluating them. Run them from the repo root, e.g.
`.venv/bin/python tools/evaluate.py report --split val`. They import paths from `src/common.py`.

**Data prep**
- `prelabel.py`: pre-label photos with a trained pill model, so labeling means correcting boxes.
- `split.py`: split our photos into train / val / test by session.
- `make_mixed_zip.py`: re-split Pill-Counter v2 by source group and build the Colab zip for the app counter.
- `make_otherpills_zip.py`: build the Colab zip of public photos for the other-medicines check.

**Training**
- `finetune.py`: two-stage fine-tuning of the v4 pill model on our photos.
- `train_classifier.py`: train the pill-form classifier on crops from `src/crops.py`.

**Evaluation**
- `evaluate.py`: tune thresholds on validation and report Levels 1 and 3–5 (the test set is guarded by `--final`).
- `compare_counters.py`: tune and compare counter models on validation, one photo per call like the app.
- `eval_mixed.py`: score counter + form classifier on the real mixed tray(s).
- `test_odd_pill.py`: paste one pill of another form into a tray and check the classifier catches it.
