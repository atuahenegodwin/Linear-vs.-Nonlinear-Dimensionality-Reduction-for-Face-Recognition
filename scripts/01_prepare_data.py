"""
Build the 46 x 56 arrays used by PCA / Kernel PCA from data/raw.zip.

Writes (float32, compressed, each file < 25 MB so it can be uploaded in the browser):
  data/processed/orl.npz            400 images, 40 people
  data/processed/yale.npz           165 images, 15 people x 11 conditions
  data/processed/eyb.npz            cleaned frontal Extended Yale B (27 people)
  data/processed/eyb_exclusions.csv images removed by the cleaning rule

Usage (from the repo root):  python scripts/01_prepare_data.py
"""
import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

RAW = None   # set by locate_raw()
OUT = "data/processed"
W, H = 46, 56
EYB_CORR_CUTOFF = 0.40      # see "Extended Yale B cleaning" below
LIT = ["normal", "happy", "sad", "sleepy", "surprised", "wink", "glasses", "noglasses"]


def locate_raw():
    """Find the folder that holds orl/, yale/ and extyaleb/ (unzip data/raw.zip if needed)."""
    global RAW
    for cand in ("data/raw", "data/raw/raw"):
        if os.path.isdir(f"{cand}/orl"):
            RAW = cand
            return
    print("extracting data/raw.zip ...")
    with zipfile.ZipFile("data/raw.zip") as z:
        z.extractall("data")                 # zip root is raw/ -> data/raw/orl, ...
    for cand in ("data/raw", "data/raw/raw"):
        if os.path.isdir(f"{cand}/orl"):
            RAW = cand
            return
    raise FileNotFoundError("could not find orl/, yale/, extyaleb/ under data/raw")


# ---------------------------------------------------------------- ORL
def prepare_orl():
    X, y = [], []
    num = lambda p: int(re.findall(r"\d+", p)[-1])
    for d in sorted(glob.glob(f"{RAW}/orl/att_faces/s*"), key=num):
        for f in sorted(glob.glob(d + "/*.pgm"), key=lambda p: int(os.path.basename(p)[:-4])):
            im = Image.open(f).convert("L").resize((W, H), Image.BOX)   # 92x112 -> 46x56
            X.append(np.asarray(im, float) / 255.0)
            y.append(num(d) - 1)
    X = np.array(X, dtype=np.float32)
    np.savez_compressed(f"{OUT}/orl.npz", images=X, labels=np.array(y))
    print("orl", X.shape, len(set(y)), "people")


# ---------------------------------------------------------------- Yale
def yale_box(imgs):
    """Crop box from the head silhouette of evenly lit photos of ONE person.
    The same box is then applied to that person's side-lit photos."""
    cx, cy = [], []
    for a in imgs:
        s = gaussian_filter(a, 2)
        s[:6, :] = 255                       # black line along the top edge
        ys, xs = np.where(s < 120)           # dark = hair/face, wall is bright
        cx.append(np.median(xs))
        cy.append(np.median(ys))
    cx, cy = np.median(cx), np.median(cy)
    h, w = 200, 164                          # aspect 46:56
    x0 = int(np.clip(cx - w / 2, 0, 320 - w))
    y0 = int(np.clip(cy - h / 2 + 5, 0, 243 - h))
    return x0, y0, x0 + w, y0 + h


def prepare_yale():
    by = {}
    for f in sorted(glob.glob(f"{RAW}/yale/yalefaces/data/subject*")):
        sid, cond = os.path.basename(f).split(".")
        by.setdefault(int(sid[7:]), {})[cond] = np.asarray(Image.open(f).convert("L"), float)
    conds = sorted({c for d in by.values() for c in d})
    X, y, c = [], [], []
    for sid in sorted(by):
        box = yale_box([by[sid][k] for k in LIT if k in by[sid]])
        for cond in conds:
            a = Image.fromarray(by[sid][cond].astype(np.uint8)).crop(box).resize((W, H), Image.BOX)
            X.append(np.asarray(a, float) / 255)
            y.append(sid - 1)
            c.append(conds.index(cond))
    X = np.array(X, dtype=np.float32)
    np.savez_compressed(f"{OUT}/yale.npz", images=X, labels=np.array(y),
                        cond=np.array(c), cond_names=np.array(conds))
    print("yale", X.shape, len(set(y)), "people,", len(conds), "conditions")


# ---------------------------------------------------------------- Extended Yale B
def prepare_eyb():
    """Frontal pose (P00), 64 lighting directions per subject, ambient shot excluded.

    Lighting subsets 1-5 use the angle between the light and the camera axis:
        <=12, <=25, <=50, <=77, >77 degrees.

    Extended Yale B cleaning: each image is correlated with the pixel-wise median of
    the SAME shot (same light direction) over all subjects; images with correlation
    below EYB_CORR_CUTOFF (misframed, eye-only crops, corrupt) are dropped.
    """
    X, y, A, E, shot = [], [], [], [], []
    for d in sorted(glob.glob(f"{RAW}/extyaleb/cropped/yaleB*")):
        sid = int(d[-2:])
        for f in sorted(glob.glob(d + "/*_P00A*.pgm")):
            m = re.search(r"A([+-]\d+)E([+-]\d+)", f)
            a, e = int(m.group(1)), int(m.group(2))
            im = Image.open(f).convert("L").resize((W, H), Image.BOX)
            X.append(np.asarray(im, float) / 255)
            y.append(sid); A.append(a); E.append(e); shot.append(f"{a:+d}_{e:+d}")
    X, y, A, E, shot = map(np.array, (X, y, A, E, shot))
    theta = np.degrees(np.arccos(np.clip(np.cos(np.radians(A)) * np.cos(np.radians(E)), -1, 1)))
    subset = np.digitize(theta, [12.0001, 25.0001, 50.0001, 77.0001]) + 1

    corr = np.zeros(len(X))
    for s in np.unique(shot):
        idx = np.where(shot == s)[0]
        flat = X[idx].reshape(len(idx), -1)
        med = np.median(flat, axis=0)
        for k, i in enumerate(idx):
            corr[i] = np.corrcoef(flat[k], med)[0, 1]
    keep = corr >= EYB_CORR_CUTOFF

    pd.DataFrame({"subject": y[~keep], "shot_A_E": shot[~keep],
                  "corr": corr[~keep].round(3)}).to_csv(f"{OUT}/eyb_exclusions.csv", index=False)
    np.savez_compressed(f"{OUT}/eyb.npz", images=X[keep].astype(np.float32), labels=y[keep],
                        A=A[keep], E=E[keep], shot=shot[keep], subset=subset[keep],
                        theta=theta[keep], corr=corr[keep])
    print(f"eyb {keep.sum()} images kept, {(~keep).sum()} excluded "
          f"({(~keep).mean() * 100:.1f}%), {len(set(y))} people")
    print("lighting subset sizes:", [int((subset[keep] == k).sum()) for k in range(1, 6)])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    locate_raw()
    prepare_orl()
    prepare_yale()
    prepare_eyb()
