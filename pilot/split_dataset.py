"""
Split a flat <ds>/images + <ds>/labels folder into train/val/test and rewrite
data.yaml. Idempotent-ish: re-running reshuffles, so only run once per dataset.

    python pilot/split_dataset.py datasets/board_v1 --ratios 0.8 0.1 0.1 --seed 0
"""
import argparse
import glob
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.ds, "images", "*.jpg")))
    if not imgs:
        raise SystemExit(f"no images in {args.ds}/images")
    random.seed(args.seed)
    random.shuffle(imgs)

    n = len(imgs)
    n_tr = int(args.ratios[0] * n)
    n_va = int(args.ratios[1] * n)
    parts = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va],
             "test": imgs[n_tr + n_va:]}

    for split, files in parts.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.ds, split, sub), exist_ok=True)
        for img in files:
            stem = os.path.splitext(os.path.basename(img))[0]
            shutil.move(img, os.path.join(args.ds, split, "images", stem + ".jpg"))
            lbl = os.path.join(args.ds, "labels", stem + ".txt")
            dst = os.path.join(args.ds, split, "labels", stem + ".txt")
            shutil.move(lbl, dst) if os.path.exists(lbl) else open(dst, "w").close()
        print(f"{split}: {len(files)}")

    for d in ("images", "labels"):
        p = os.path.join(args.ds, d)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)

    open(os.path.join(args.ds, "data.yaml"), "w").write(
        f"path: {os.path.abspath(args.ds)}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: logo\n")
    print(f"wrote {args.ds}/data.yaml")


if __name__ == "__main__":
    main()
