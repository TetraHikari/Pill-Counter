"""Tune thresholds on val, then compare all levels by count accuracy.

    python tools/evaluate.py tune --model results/models/v4_pill_model.pt --name v4
    python tools/evaluate.py tune --model results/models/thai_v2.pt --name thai_v2
    python tools/evaluate.py report --split val            # sanity check, as often as you like
    python tools/evaluate.py report --split test --final   # the one real test run

Levels: 1 = predict the mean train count, 3 = COCO-pretrained YOLO (no pill training),
4 = v4 pill model, 5 = v4 + fine-tuned on our photos. (The report's Level 2 baseline is not part of
this repository.) Levels 4 and 5 use the thresholds tuned on
val. Confidence intervals resample whole sessions, because photos of one pile are not independent.
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import MODELS, REAL, RESULTS, device, load_split

IMGSZ = 1024
LEVELS = [  # (name, weights, thresholds file or None for fixed defaults)
    ("L3 pretrained YOLO (COCO)", "yolo11s.pt", None),
    ("L4 v4 pill model", MODELS / "v4_pill_model.pt", "v4"),
    ("L5 v4 + Thai fine-tune (thai_v2)", MODELS / "thai_v2.pt", "thai_v2"),
    ("L5 thai_v3_m1024 (YOLO11m from COCO)", MODELS / "thai_v3_m1024.pt", "thai_v3_m1024_compare"),
    ("L5 app counter (thai_mix_s1024, ours + public multi-drug)", MODELS / "thai_mix_s1024.pt", "thai_mix_s1024_single"),
]


def box_confs(model, photos, iou):
    """Confidence of every box per photo, at a low cut-off so any higher cut-off can be applied later."""
    # One photo per call, as the app does: given a list with mixed image sizes, Ultralytics letterboxes every
    # photo to a padded square and the counts change.
    return [model.predict(str(REAL / "images" / f), imgsz=IMGSZ, conf=0.05, iou=iou, max_det=500,
                          device=device(), verbose=False)[0].boxes.conf.cpu().numpy() for f in photos]


def tune(args):
    df = load_split()
    val = df[df.split == "val"]
    truth = val.true_count.to_numpy()
    model = YOLO(args.model)
    grid = []
    for iou in [0.4, 0.5, 0.6, 0.7]:
        confs = box_confs(model, list(val.photo), iou)
        for c in np.round(np.arange(0.10, 0.90, 0.05), 2):
            pred = np.array([(x >= c).sum() for x in confs])
            grid.append(dict(iou=iou, conf=float(c), exact=float(np.mean(pred == truth)),
                             mae=float(np.mean(np.abs(pred - truth)))))
    g = pd.DataFrame(grid)
    # Lowest MAE, then best exact rate, then the cut-off closest to 0.5. MAE first because val is
    # tiny: exact rate on a handful of photos jumps around, MAE is far steadier.
    g["dist"] = (g.conf - 0.5).abs()
    order = dict(by=["mae", "exact", "dist"], ascending=[True, False, True])
    best = g.sort_values(**order).iloc[0]
    out = dict(model=str(args.model), conf=best.conf, iou=best.iou, val_exact=best.exact,
               val_mae=best.mae, val_photos=len(val), val_sessions=sorted(val.session_id.unique()))
    (RESULTS / f"thresholds_{args.name}.json").write_text(json.dumps(out, indent=2))
    print(g.sort_values(**order).head(8).drop(columns="dist").to_string(index=False))
    print(f"\nSaved results/thresholds_{args.name}.json: conf={best.conf} iou={best.iou}")


def metrics(d):
    err = d.pred - d.true_count
    return dict(n=len(d), exact=np.mean(err == 0), mae=np.mean(np.abs(err)),
                rel=np.mean(np.abs(err) / d.true_count), bias=np.mean(err))


def session_ci(d, key, n_boot=2000, seed=0):
    """95% CI by resampling sessions (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("session_id")]
    if len(groups) < 2:
        return (np.nan, np.nan)
    vals = [metrics(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]))[key]
            for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def report(args):
    df = load_split()
    if args.split == "test":
        log = RESULTS / "test_log.txt"
        if not args.final:
            raise SystemExit("The test set is used once. Re-run with --final when every choice is frozen.")
        if log.exists() and not args.again:
            raise SystemExit(f"Test set already used (see {log}). Use --again only if you must, and say so in the report.")
        with open(log, "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} test run\n")

    d = df[df.split == args.split].copy()
    rows = []
    mean_train = round(df[df.split == "train"].true_count.mean())
    rows.append(d.assign(level="L1 mean train count", pred=mean_train))
    for name, weights, th in LEVELS:
        if th and not (RESULTS / f"thresholds_{th}.json").exists():
            print(f"Skipping {name}: run `evaluate.py tune --name {th}` first")
            continue
        t = json.loads((RESULTS / f"thresholds_{th}.json").read_text()) if th else dict(conf=0.25, iou=0.7)
        confs = box_confs(YOLO(str(weights)), list(d.photo), t["iou"])
        rows.append(d.assign(level=f"{name} (conf={t['conf']}, iou={t['iou']})",
                             pred=[int((x >= t["conf"]).sum()) for x in confs]))
    preds = pd.concat(rows)
    preds.to_csv(RESULTS / f"{args.split}_predictions.csv", index=False)

    lines = [f"# Count accuracy on {args.split} ({d.photo.nunique()} photos, {d.session_id.nunique()} sessions)", "",
             "CIs resample whole sessions; with few sessions they are wide, which is the honest answer.", "",
             "| Level | Photos | Exact count rate [95% CI] | MAE [95% CI] | Rel. error | Bias |",
             "|---|---|---|---|---|---|"]
    for cond in ["all", "seen", "unseen"]:
        sub = preds if cond == "all" else preds[preds.condition == cond]
        if sub.empty:
            continue
        lines.append(f"| **{cond} conditions** | | | | | |")
        for lvl, g in sub.groupby("level", sort=False):
            m = metrics(g); ce = session_ci(g, "exact"); cm = session_ci(g, "mae")
            lines.append(f"| {lvl} | {m['n']} | {m['exact']:.0%} [{ce[0]:.0%}, {ce[1]:.0%}] | "
                         f"{m['mae']:.2f} [{cm[0]:.2f}, {cm[1]:.2f}] | {m['rel']:.1%} | {m['bias']:+.2f} |")
    lines += ["", "## Per session (predicted counts for each photo)", "", "| Session | True | Condition | "
              + " | ".join(l.split(" (")[0] for l in preds.level.unique()) + " |",
              "|---|---|---|" + "---|" * preds.level.nunique()]
    for sid, g in d.groupby("session_id"):
        cells = [str(list(preds[(preds.session_id == sid) & (preds.level == l)].pred)) for l in preds.level.unique()]
        lines.append(f"| {sid} {g.background.iloc[0]} | {g.true_count.iloc[0]} | {g.condition.iloc[0]} | "
                     + " | ".join(cells) + " |")
    out = RESULTS / f"{args.split}_results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out} and {args.split}_predictions.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--model", required=True)
    t.add_argument("--name", required=True)
    t.set_defaults(fn=tune)
    r = sub.add_parser("report")
    r.add_argument("--split", choices=["val", "test"], default="val")
    r.add_argument("--final", action="store_true")
    r.add_argument("--again", action="store_true")
    r.set_defaults(fn=report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
