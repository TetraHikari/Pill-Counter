"""Score counter + form classifier on the real mixed tray(s) in data/mixed_trays.csv.

    python tools/eval_mixed.py --counter thai_v2.pt --thresholds thai_v2 --classifier pill_type.pt

Mixed trays are never used for training or choosing models. Scores are per photo, against the
hand-counted totals: total count, and the count of each form.
"""
import argparse
import glob
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import MODELS, REAL, RESULTS, ROOT, device
from crops import square_crop

FORMS = ["tablet", "caplet", "mini_tablet"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counter", default="thai_v2.pt")
    p.add_argument("--thresholds", default="thai_v2")
    p.add_argument("--classifier", default="pill_type.pt")
    p.add_argument("--trays", nargs="+", help="only these sessions (default: all in mixed_trays.csv)")
    args = p.parse_args()
    det, cls = YOLO(str(MODELS / args.counter)), YOLO(str(MODELS / args.classifier))
    t = json.loads((RESULTS / f"thresholds_{args.thresholds}.json").read_text())
    trays = pd.read_csv(ROOT / "data" / "mixed_trays.csv")
    if args.trays:
        trays = trays[trays.session_id.isin(args.trays)]
    for tray in trays.itertuples():
        true = np.array([tray.true_count] + [getattr(tray, f) for f in FORMS])
        rows = []
        for f in sorted(glob.glob(str(REAL / "images" / f"{tray.session_id}_*.jpg"))):
            img = cv2.imread(f); H, W = img.shape[:2]
            r = det.predict(img, imgsz=t.get("imgsz", 1024), conf=t["conf"], iou=t["iou"], max_det=500, device=device(), verbose=False)[0]
            crops = [square_crop(img, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
                     for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]
            c = Counter(cls.names[int(q.probs.top1)] for q in cls.predict(crops, imgsz=224, verbose=False)) if crops else Counter()
            rows.append([len(r.boxes)] + [c[f] for f in FORMS])
        e = np.abs(np.array(rows) - true)
        print(f"{tray.session_id} ({len(rows)} photos; true {tray.true_count} = " + " / ".join(f"{getattr(tray, f)} {f}" for f in FORMS) + ")")
        print(f"  counter {args.counter} + classifier {args.classifier}")
        for k, name in enumerate(["TOTAL"] + FORMS):
            print(f"  {name:12s} exact {np.mean(e[:, k] == 0):4.0%} of photos   average miss {e[:, k].mean():.1f}")
        print(f"  all three forms exact in one photo: {np.mean((e[:, 1:] == 0).all(1)):.0%}")


if __name__ == "__main__":
    main()
