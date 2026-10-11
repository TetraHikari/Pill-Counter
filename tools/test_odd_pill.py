"""Can the pill-type classifier catch ONE wrong pill in a tray of another type?

    python tools/test_odd_pill.py

Takes held-out photos (val/test sessions only), pastes a pill of the other type over one existing
pill, then runs the same crop + mask + classify step the app uses on that box. A classifier that
leans on the background or neighbouring pills fails this test even when its normal accuracy is 100%.
"""
import itertools
import random
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import REAL, MODELS, SESSIONS
from crops import square_crop

N = 100
# held-out sessions per pill type (never used to train the classifier); add caplet / mini tablet when they exist
TRAYS = {"tablet": ["s03", "s04", "s08", "s24"], "caplet": ["s14"], "mini_tablet": ["s19"]}


def boxes_of(photo):
    return np.loadtxt(REAL / "labels" / f"{photo[:-4]}.txt", ndmin=2)[:, 1:]


def cut_disc(img, b, scale=0.95):
    """The pill itself: a disc cut from its box, plus its alpha mask."""
    H, W = img.shape[:2]
    cx, cy, w, h = b[0] * W, b[1] * H, b[2] * W, b[3] * H
    r = int(min(w, h) / 2 * scale)
    patch = img[int(cy) - r:int(cy) + r, int(cx) - r:int(cx) + r].copy()
    mask = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    return patch, mask


def main():
    random.seed(0)
    if len(TRAYS) < 2:
        raise SystemExit("Needs held-out sessions of at least two pill types in TRAYS.")
    model = YOLO(str(MODELS / "pill_type.pt"))
    df = pd.read_csv(SESSIONS)
    for tray_type, odd_type in itertools.permutations(TRAYS, 2):
        trays = df[df.session_id.isin(TRAYS[tray_type])].photo.tolist()
        donors = df[df.session_id.isin(TRAYS[odd_type])].photo.tolist()
        crops, neighbour_crops = [], []
        for _ in range(N):
            tray_photo, donor_photo = random.choice(trays), random.choice(donors)
            tray, donor = cv2.imread(str(REAL / "images" / tray_photo)), cv2.imread(str(REAL / "images" / donor_photo))
            tb, db = boxes_of(tray_photo), boxes_of(donor_photo)
            i = random.randrange(len(tb))
            patch, mask = cut_disc(donor, db[random.randrange(len(db))])
            H, W = tray.shape[:2]
            cx, cy, w, h = tb[i][0] * W, tb[i][1] * H, tb[i][2] * W, tb[i][3] * H
            d = int(min(w, h) * 0.95)
            patch, mask = cv2.resize(patch, (d, d)), cv2.resize(mask, (d, d))
            x0, y0 = int(cx - d / 2), int(cy - d / 2)
            if x0 < 0 or y0 < 0 or x0 + d > W or y0 + d > H:
                continue
            roi = tray[y0:y0 + d, x0:x0 + d]
            roi[mask > 0] = patch[mask > 0]
            crops.append(square_crop(tray, *tb[i]))
            j = (i + 1) % len(tb)                     # an untouched neighbour, should keep the tray's type
            neighbour_crops.append(square_crop(tray, *tb[j]))
        odd = Counter(model.names[int(r.probs.top1)] for r in model.predict(crops, imgsz=224, verbose=False))
        nb = Counter(model.names[int(r.probs.top1)] for r in model.predict(neighbour_crops, imgsz=224, verbose=False))
        print(f"One {odd_type} pasted into a {tray_type} tray ({len(crops)} cases):")
        print(f"   odd pill   -> {dict(odd)}   caught: {odd[odd_type] / len(crops):.0%}")
        print(f"   neighbours -> {dict(nb)}   still correct: {nb[tray_type] / len(neighbour_crops):.0%}")


if __name__ == "__main__":
    main()
