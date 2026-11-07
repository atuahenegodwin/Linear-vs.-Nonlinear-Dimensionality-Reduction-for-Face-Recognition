"""
Face data and train/test splits shared by the CCA and gauge scripts.

Reads the team's data/raw.zip directly (no need to unzip).
Preprocessing: grayscale -> crop to the 46:56 aspect ratio -> resize to
46 x 56 = 2,576 pixels -> scale to [0, 1].
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]           # repository root
DEFAULT_ZIP = ROOT / "data" / "raw.zip"
SIZE = (46, 56)          # width x height, as in the proposal
YALE_CONDITIONS = ["centerlight", "glasses", "happy", "leftlight", "noglasses", "normal",
                   "rightlight", "sad", "sleepy", "surprised", "wink"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def to_vector(data):
    """Bytes of one image file -> 2,576 pixel values in [0, 1]."""
    img = Image.open(io.BytesIO(data)).convert("L")
    w, h = img.size
    target = SIZE[0] / SIZE[1]
    if w / h > target:                  # too wide: crop the sides
        new_w = round(h * target)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif w / h < target:                # too tall: crop top and bottom
        new_h = round(w / target)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize(SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64).ravel() / 255.0


def load_all(zip_path):
    """Return a dict of datasets, each with X, person ids and extra labels."""
    orl, yale, eyb = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or name.endswith("/") or ".DS_Store" in name:
                continue
            m = re.search(r"att_faces/s(\d+)/\d+\.pgm$", name)
            if m:
                orl.append((int(m.group(1)), to_vector(z.read(name))))
                continue
            m = re.search(r"yalefaces/data/subject(\d+)\.(\w+)$", name)
            if m and m.group(2) in YALE_CONDITIONS:
                yale.append((int(m.group(1)), m.group(2), to_vector(z.read(name))))
                continue
            m = re.search(r"cropped/yaleB(\d+)/yaleB\d+_P00A([+-]\d+)E([+-]\d+)\.pgm$", name)
            if m:   # the "Ambient" (no light) images do not match and are skipped
                az, el = int(m.group(2)), int(m.group(3))
                angle = np.degrees(np.arccos(np.cos(np.radians(az)) * np.cos(np.radians(el))))
                eyb.append((int(m.group(1)), angle, to_vector(z.read(name))))

    def subset(a):   # standard Yale B lighting subsets
        return 1 if a <= 12 else 2 if a <= 25 else 3 if a <= 50 else 4 if a <= 77 else 5

    return {
        "ORL": dict(X=np.array([v for _, v in orl]), y=np.array([p for p, _ in orl])),
        "Yale": dict(X=np.array([v for *_, v in yale]), y=np.array([p for p, *_ in yale]),
                     cond=np.array([c for _, c, _ in yale])),
        "ExtYaleB": dict(X=np.array([v for *_, v in eyb]), y=np.array([p for p, *_ in eyb]),
                         subset=np.array([subset(a) for _, a, _ in eyb])),
    }


# ----------------------------------------------------------------------
# Splits: each yields (setting name, split number, train index, test index)
# ----------------------------------------------------------------------
def per_person_split(y, n_train, rng):
    """Stratified split: pick n_train random photos of EVERY person for training.

    Splitting per person (not from the whole pile) guarantees that every person
    is in the training map, so the classifier has seen every name it may need
    to predict. The remaining photos of each person go to testing.
    """
    train = []
    for p in np.unique(y):
        idx = np.flatnonzero(y == p)
        train.extend(rng.choice(idx, n_train, replace=False))
    train = np.array(sorted(train))
    return train, np.setdiff1d(np.arange(len(y)), train)


def all_splits(data, n_random):
    rng = np.random.default_rng(0)
    for s in range(n_random):
        yield ("ORL", "ORL", s, *per_person_split(data["ORL"]["y"], 5, rng))
    cond = data["Yale"]["cond"]
    for s, c in enumerate(YALE_CONDITIONS):
        yield ("Yale", "Yale", s, np.flatnonzero(cond != c), np.flatnonzero(cond == c))
    rng = np.random.default_rng(0)
    for s in range(min(n_random, 5)):
        yield ("ExtYaleB", "ExtYaleB random", s, *per_person_split(data["ExtYaleB"]["y"], 30, rng))
    sub = data["ExtYaleB"]["subset"]
    for k in (2, 3, 4, 5):
        yield ("ExtYaleB", f"ExtYaleB light: test subset {k}", 0,
               np.flatnonzero(sub == 1), np.flatnonzero(sub == k))
