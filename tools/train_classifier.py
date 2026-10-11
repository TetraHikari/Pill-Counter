"""Train the pill-type classifier on crops from crops.py (option B: detect, then classify).

    python tools/train_classifier.py
    python tools/train_classifier.py --eval-only     # just score the saved model on the test crops

Starts from yolo11n-cls.pt (pretrained on ImageNet) and fine-tunes it on data/crops. Saves
results/models/pill_type.pt and prints accuracy per class on the held-out test sessions.
"""
import argparse
import shutil
from collections import Counter

from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import MODELS, RESULTS, ROOT, device

CROPS = ROOT / "data" / "crops"
OUT = MODELS / "pill_type.pt"


def evaluate(model_path):
    m = YOLO(str(model_path))
    names = m.names
    confusion = Counter()
    for folder in sorted((CROPS / "test").iterdir()):
        for r in m.predict(sorted(map(str, folder.glob("*.jpg"))), imgsz=224, device=device(),
                           stream=True, verbose=False):
            confusion[(folder.name, names[int(r.probs.top1)])] += 1
    classes = sorted({t for t, _ in confusion} | {p for _, p in confusion})
    print("\nTest crops (held-out test sessions), rows = true, columns = predicted")
    print(" " * 14 + "".join(f"{c:>14s}" for c in classes))
    for t in classes:
        row = [confusion[(t, p)] for p in classes]
        if sum(row):
            print(f"{t:14s}" + "".join(f"{v:14d}" for v in row) + f"   accuracy {confusion[(t, t)] / sum(row):.1%}")
    total = sum(confusion.values())
    print(f"overall accuracy {sum(confusion[(c, c)] for c in classes) / total:.1%} on {total} crops")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    classes = sorted(d.name for d in (CROPS / "train").iterdir() if d.is_dir()) if (CROPS / "train").exists() else []
    if len(classes) < 2:
        raise SystemExit(f"Need at least two pill types to train a classifier, found {classes}. "
                         "Photograph another form (caplet, mini tablet), set its pill_type in sessions.csv, run crops.py.")
    if not args.eval_only:
        model = YOLO("yolo11n-cls.pt")
        model.train(data=str(CROPS), imgsz=224, epochs=args.epochs, batch=32, device=device(),
                    project=str(RESULTS / "runs"), name="pill_type", exist_ok=True, patience=10,
                    degrees=180, flipud=0.5, fliplr=0.5, hsv_v=0.4, hsv_s=0.2, hsv_h=0.0)  # colour is the label: little colour jitter
        MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copy(model.trainer.save_dir / "weights" / "best.pt", OUT)
        print(f"Saved {OUT}")
    evaluate(OUT)


if __name__ == "__main__":
    main()
