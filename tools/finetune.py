"""Fine-tune the v4 pill model on our own Thai-medicine photos (Level 5).

    python tools/finetune.py                      # v4 pill model -> thai model
    python tools/finetune.py --weights yolo11s.pt --name no_v4   # ablation: skip the v4 stage

Stage 1 trains with the backbone frozen, so 30 photos can't wreck what v4 taught it.
Stage 2 unfreezes everything at a low learning rate. Uses train.txt / val.txt from split.py.
"""
import argparse
import shutil

from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import MODELS, RESULTS, data_yaml, device

AUG = dict(degrees=180, flipud=0.5, fliplr=0.5, scale=0.3, hsv_v=0.4, hsv_s=0.5, translate=0.1)
NO_AUG = dict(degrees=0, flipud=0, fliplr=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0, translate=0, mosaic=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(MODELS / "v4_pill_model.pt"))
    p.add_argument("--name", default="thai")
    p.add_argument("--epochs1", type=int, default=30)
    p.add_argument("--epochs2", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--no-aug", action="store_true")
    args = p.parse_args()

    common = dict(data=str(data_yaml()), imgsz=args.imgsz, batch=args.batch, device=device(),
                  project=str(RESULTS / "runs"), exist_ok=True, max_det=500, optimizer="SGD",
                  patience=25, workers=2, **(NO_AUG if args.no_aug else AUG))

    stage1 = YOLO(args.weights)
    stage1.train(epochs=args.epochs1, freeze=10, lr0=0.005, name=f"{args.name}_stage1", **common)
    best1 = stage1.trainer.save_dir / "weights" / "best.pt"

    stage2 = YOLO(best1)
    stage2.train(epochs=args.epochs2, lr0=0.001, warmup_epochs=0, name=f"{args.name}_stage2", **common)
    best2 = stage2.trainer.save_dir / "weights" / "best.pt"

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / f"{args.name}.pt"
    shutil.copy(best2, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
