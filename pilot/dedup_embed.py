"""
Collapse a big pile of harvested crops to a manageable set of visually-distinct
representatives, using ImageNet CNN features + clustering. Pixel hashing does not
work here (every LED frame differs slightly); feature-space clustering groups
"the same creative seen many times" into one.

    python pilot/dedup_embed.py datasets/ads_db_raw --k 450 --out datasets/ads_db_reps

Writes <out>/crops/  (one medoid per cluster) + <out>/_meta.jsonl (carried over)
+ <out>/_clusters.json (cluster -> member files, sizes). Feed <out> to
pilot/presort_crops.py or sort it by hand.
"""

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="dir with crops/ + _meta.jsonl")
    ap.add_argument("--k", type=int, default=450, help="number of representatives")
    ap.add_argument("--out", default="datasets/ads_db_reps")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    import torch
    from torchvision import models, transforms

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 256), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    files = sorted(glob.glob(os.path.join(args.raw, "crops", "*.jpg")))
    print(f"{len(files)} crops -> embedding on {dev}")
    feats = []
    buf, idx = [], []
    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            f = net(x).cpu().numpy()
        f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        feats.append(f)
        buf.clear()
    for i, p in enumerate(files):
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        buf.append(tf(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        idx.append(p)
        if len(buf) >= args.batch:
            flush()
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(files)}")
    flush()
    X = np.vstack(feats)
    print(f"features {X.shape}")

    from sklearn.cluster import MiniBatchKMeans
    k = min(args.k, len(idx))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=1024)
    lab = km.fit_predict(X)

    # medoid = crop closest to its cluster centroid
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    meta_in = {}
    mp = os.path.join(args.raw, "_meta.jsonl")
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                meta_in[r["file"]] = r
    clusters = {}
    mo = open(os.path.join(args.out, "_meta.jsonl"), "w", encoding="utf-8")
    for c in range(k):
        mask = np.where(lab == c)[0]
        if not len(mask):
            continue
        d = np.linalg.norm(X[mask] - km.cluster_centers_[c], axis=1)
        best = idx[mask[d.argmin()]]
        bn = os.path.basename(best)
        shutil.copy2(best, os.path.join(args.out, "crops", bn))
        clusters[str(c)] = {"rep": bn, "n": int(len(mask)),
                            "members": [os.path.basename(idx[m]) for m in mask]}
        if bn in meta_in:
            r = dict(meta_in[bn]); r["cluster_size"] = int(len(mask))
            mo.write(json.dumps(r, ensure_ascii=False) + "\n")
    mo.close()
    json.dump(clusters, open(os.path.join(args.out, "_clusters.json"), "w"),
              ensure_ascii=False, indent=1)
    sizes = sorted((v["n"] for v in clusters.values()), reverse=True)
    print(f"\n{len(clusters)} representatives -> {args.out}/crops")
    print(f"cluster sizes: max {sizes[0]}, median {sizes[len(sizes)//2]}, "
          f"singletons {sum(1 for s in sizes if s == 1)}")


if __name__ == "__main__":
    main()
