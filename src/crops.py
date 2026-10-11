"""Cut every labeled pill out of the full-resolution photos, for the pill-type classifier (option B).

    python src/crops.py

Writes data/crops/{train,val,test}/<pill_type>/<photo>_<n>.jpg (Ultralytics classification layout).
Classes come from the pill_type column of data/sessions.csv, so adding a drug = adding sessions.

Split by session, the same held-out sessions as the detector (split.py), so neither model
trains on a session the other is tested on. The blue tablets (s03, s04) are only in test, which
also checks that 'tablet' is recognised by form, not colour. The mixed tray (s20) is not in
sessions.csv: its pills have no per-pill form label, so it is only used for evaluation.
"""
import shutil

import cv2
import numpy as np
import pandas as pd

from common import REAL, ROOT, SESSIONS

OUT = ROOT / "data" / "crops"
PAD = 0.15        # extra margin around each box, as a fraction of its size
SIZE = 224        # classifier input size
TEST = {"s03", "s04", "s08", "s14", "s19", "s24"}
VAL_SESSIONS = {"s06", "s13", "s18", "s23"}
VAL_PHOTOS = set()
JITTER_COPIES = 2   # extra crops per TRAIN pill with a randomly resized/shifted box (counter boxes vary a bit)
JITTER_SCALE = (0.85, 1.2)
JITTER_SHIFT = 0.1  # fraction of box width/height


def square_crop(img, cx, cy, w, h):
    """Square crop centred on the box, showing ONLY the pill: everything outside the ellipse
    inscribed in the box is plain gray. Without the mask the classifier learned the background
    and neighbouring pills (a blue pill among white ones was called white 89% of the time).
    cx, cy, w, h are fractions of the image size (YOLO format). Used for training and in the app."""
    H, W = img.shape[:2]
    bw, bh = w * W, h * H
    side = int(max(bw, bh) * (1 + 2 * PAD))
    x0, y0 = int(cx * W - side / 2), int(cy * H - side / 2)
    out = np.full((side, side, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    mask = np.zeros((side, side), np.uint8)
    cv2.ellipse(mask, (side // 2, side // 2), (int(bw / 2 * 0.82), int(bh / 2 * 0.82)), 0, 0, 360, 255, -1)
    out[mask == 0] = 114
    return cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def split_of(row):
    if row.session_id in TEST:
        return "test"
    if row.session_id in VAL_SESSIONS or row.photo in VAL_PHOTOS:
        return "val"
    return "train"


def jitter(cx, cy, w, h, rng):
    """A box like the counter might draw it: slightly bigger/smaller and off-centre."""
    sw, sh = rng.uniform(*JITTER_SCALE, size=2)
    dx, dy = rng.uniform(-JITTER_SHIFT, JITTER_SHIFT, size=2)
    return cx + dx * w, cy + dy * h, w * sw, h * sh


def main():
    rng = np.random.default_rng(0)
    if OUT.exists():
        shutil.rmtree(OUT)
    df = pd.read_csv(SESSIONS)
    df = df[df.pill_type != "mixed"]   # mixed trays have no per-pill form label: counter only
    counts = {}
    for row in df.itertuples():
        img = cv2.imread(str(REAL / "images" / row.photo))
        labels = np.loadtxt(REAL / "labels" / f"{row.photo[:-4]}.txt", ndmin=2)
        split = split_of(row)
        folder = OUT / split / row.pill_type
        folder.mkdir(parents=True, exist_ok=True)
        for n, (_, cx, cy, w, h) in enumerate(labels):
            cv2.imwrite(str(folder / f"{row.photo[:-4]}_{n:03d}.jpg"), square_crop(img, cx, cy, w, h),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if split == "train":   # val/test keep the exact hand-checked boxes
                for k in range(JITTER_COPIES):
                    cv2.imwrite(str(folder / f"{row.photo[:-4]}_{n:03d}_j{k}.jpg"),
                                square_crop(img, *jitter(cx, cy, w, h, rng)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        counts[(split, row.pill_type)] = counts.get((split, row.pill_type), 0) + len(labels) * (1 + JITTER_COPIES * (split == "train"))
    for (split, cls), n in sorted(counts.items()):
        print(f"{split:5s} {cls:12s} {n:5d} crops")


if __name__ == "__main__":
    main()
