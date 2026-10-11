"""Tune and compare counter models on validation, the same way for each.

    python tools/compare_counters.py thai_v2 thai_v3_s1024 thai_v3_s1280 thai_v3_m1024
    python tools/compare_counters.py --rule both thai_v2 thai_mix_s1024

For each model: run it on the val photos once per NMS IoU (at a low cut-off), try every confidence
cut-off, keep the setting with the lowest average miss (then best exact rate, then cut-off closest
to 0.5). --rule picks how the average is taken: "photo" (default; every photo counts once) or
"session" (each val session counts equally, so the 30-photo s26 does not outweigh the rest); "both"
runs the two from the same predictions. Saves results/thresholds_<name>_compare.json (photo rule) and
results/thresholds_<name>_compare_session.json (session rule); never overwrites the file the app uses.
Choose the model from this table only; the test sessions and mixed tray s20 are not used here.
"""
import argparse
import json

import numpy as np
import torch
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import MODELS, REAL, RESULTS, device, load_split

SUFFIX = {"photo": "", "session": "_session"}


def scores(pred, truth, sess):
    """(photo-weighted exact, MAE), (session-weighted exact, MAE), per-session {sid: (exact, MAE)}."""
    err = pred - truth
    per = {sid: (float(np.mean(err[sess == sid] == 0)), float(np.mean(np.abs(err[sess == sid]))))
           for sid in sorted(set(sess))}
    photo = (float(np.mean(err == 0)), float(np.mean(np.abs(err))))
    session = (float(np.mean([e for e, _ in per.values()])), float(np.mean([a for _, a in per.values()])))
    return photo, session, per


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+")
    p.add_argument("--rule", choices=["photo", "session", "both"], default="photo")
    args = p.parse_args()
    rules = ["photo", "session"] if args.rule == "both" else [args.rule]
    df = load_split()
    val = df[df.split == "val"].reset_index(drop=True)
    truth, sess = val.true_count.to_numpy(), val.session_id.to_numpy()
    paths = [str(REAL / "images" / p) for p in val.photo]
    summary = []
    for name in args.names:
        imgsz = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=False)["train_args"]["imgsz"]
        m = YOLO(str(MODELS / f"{name}.pt"))
        best = dict.fromkeys(rules)
        for iou in [0.4, 0.5, 0.6, 0.7]:
            # One photo per call, as the app does: given a list with mixed image sizes (s26 has 5712x4284 and
            # 4032x3024 photos), Ultralytics letterboxes every photo to a padded square and the counts change.
            confs = [m.predict(p, imgsz=imgsz, conf=0.05, iou=iou, max_det=500, device=device(), verbose=False)[0]
                     .boxes.conf.cpu().numpy() for p in paths]
            for c in np.round(np.arange(0.10, 0.90, 0.05), 2):
                pred = np.array([(x >= c).sum() for x in confs])
                photo, session, _ = scores(pred, truth, sess)
                for rule, (ex, mae) in zip(["photo", "session"], [photo, session]):
                    key = (mae, -ex, abs(c - 0.5))
                    if rule in best and (best[rule] is None or key < best[rule][0]):
                        best[rule] = (key, float(c), iou, pred)
        for rule in rules:
            _, conf, iou, pred = best[rule]
            photo, session, per = scores(pred, truth, sess)
            (RESULTS / f"thresholds_{name}_compare{SUFFIX[rule]}.json").write_text(json.dumps(
                dict(model=f"{name}.pt", conf=conf, iou=iou, imgsz=int(imgsz), rule=rule,
                     val_exact=photo[0], val_mae=photo[1], val_exact_session=session[0], val_mae_session=session[1],
                     val_sessions=sorted(per)), indent=2))
            summary.append((rule, name, imgsz, conf, iou, photo, session, per))
    print(f"VALIDATION ({len(val)} photos, sessions {', '.join(sorted(val.session_id.unique()))})")
    for rule, name, imgsz, conf, iou, photo, session, per in summary:
        sess_txt = " | ".join(f"{sid} {e:.0%}/{a:.1f}" for sid, (e, a) in per.items())
        print(f"  [{rule:7s}] {name:15s} {imgsz:4d}px conf {conf:.2f} iou {iou:.1f} | photo-w exact {photo[0]:4.0%} "
              f"MAE {photo[1]:.2f} | session-w exact {session[0]:4.0%} MAE {session[1]:.2f} | {sess_txt}")


if __name__ == "__main__":
    main()
