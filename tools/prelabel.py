"""Pre-label photos with a trained pill model, so labeling means correcting boxes, not drawing them.

    python tools/prelabel.py data/real/images --weights results/models/v4_pill_model.pt

Writes one YOLO .txt per image into ../labels (next to the images folder), plus data.yaml and
classes.txt, so the folder can be uploaded to Roboflow (or opened in CVAT / X-AnyLabeling) with
the boxes already in place. Existing label files are kept unless --overwrite is given, so
corrected labels are never clobbered.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def device():
    import torch
    if torch.cuda.is_available():
        return 0
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, help="folder of photos")
    p.add_argument("--weights", default="results/models/v4_pill_model.pt")
    p.add_argument("--labels", type=Path, help="output folder (default: sibling 'labels' folder)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    labels = args.labels or args.images.parent / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / "classes.txt").write_text("pill\n")
    (args.images.parent / "data.yaml").write_text("names:\n  0: pill\nnc: 1\n")

    todo = [f for f in sorted(args.images.iterdir())
            if f.suffix.lower() in IMG_EXTS and (args.overwrite or not (labels / f"{f.stem}.txt").exists())]
    if not todo:
        print("Nothing to do: every image already has a label file (use --overwrite to redo).")
        return

    model = YOLO(args.weights)
    total = 0
    for r in model.predict(todo, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=500,
                           device=device(), stream=True, verbose=False):
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in r.boxes.xywhn.tolist()]
        (labels / f"{Path(r.path).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
        print(f"{Path(r.path).name:30s} {len(lines):4d} boxes")
    print(f"\nPre-labeled {len(todo)} images ({total} boxes) -> {labels}")


if __name__ == "__main__":
    main()
