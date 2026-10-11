"""Split our photos by session, so photos of the same pile never sit in two splits.

    python tools/split.py --val s06 s13 s18 s23 s26 --test s03 s04 s08 s14 s19 s24

Sessions not named in --val or --test go to train. A val/test photo is tagged "unseen" when its
background or lighting never appears in train (Experiment 2 in the proposal).
Writes data/real/{train,val,test}.txt and data/real/split.csv.
"""
import argparse

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # common.py (paths), crops.py

from common import REAL, SESSIONS, SPLITS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", nargs="+", default=["s06", "s13", "s18", "s23", "s26"])
    p.add_argument("--test", nargs="+", default=["s03", "s04", "s08", "s14", "s19", "s24"])
    args = p.parse_args()

    df = pd.read_csv(SESSIONS)
    unknown = (set(args.val) | set(args.test)) - set(df.session_id)
    if unknown:
        raise SystemExit(f"Unknown sessions: {sorted(unknown)}")
    if set(args.val) & set(args.test):
        raise SystemExit("A session can't be in both val and test.")

    df["split"] = "train"
    df.loc[df.session_id.isin(args.val), "split"] = "val"
    df.loc[df.session_id.isin(args.test), "split"] = "test"

    missing = [f for f in df.photo if not (REAL / "labels" / f"{f[:-4]}.txt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} photos have no label file, e.g. {missing[:3]}")

    train = df[df.split == "train"]
    seen_bg, seen_light = set(train.background), set(train.lighting)
    df["condition"] = ["unseen" if (s != "train" and (b not in seen_bg or l not in seen_light)) else "seen"
                       for s, b, l in zip(df.split, df.background, df.lighting)]

    for s in SPLITS:
        (REAL / f"{s}.txt").write_text("".join(f"./images/{f}\n" for f in df[df.split == s].photo))
    df[["photo", "split", "condition"]].to_csv(REAL / "split.csv", index=False)

    for s in SPLITS:
        d = df[df.split == s]
        sessions = ", ".join(f"{sid}({g.true_count.iloc[0]} {g.background.iloc[0]}"
                             f"{', UNSEEN' if (g.condition == 'unseen').any() else ''})"
                             for sid, g in d.groupby("session_id"))
        print(f"{s:5s} {len(d):3d} photos  {d.session_id.nunique()} sessions: {sessions}")


if __name__ == "__main__":
    main()
