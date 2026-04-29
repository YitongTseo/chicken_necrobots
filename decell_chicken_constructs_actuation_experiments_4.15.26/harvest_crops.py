"""Harvest additional display crops for labeling.

Samples N_PER_VIDEO evenly-spaced frames from each .mov, crops the Force Out
display region, pre-labels with EasyOCR, and saves to digit_crops/to_label/
with filename {label}_{tag}_{idx:03d}.png. Skips indices that already exist.
"""
from pathlib import Path
import cv2
import numpy as np
import easyocr

HERE = Path(__file__).parent
DATA_DIR = HERE / "actuation_currents_data"
LABEL_DIR = HERE / "digit_crops" / "to_label"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

N_PER_VIDEO = 80
DECIMAL_FROM_RIGHT = 3

EXPERIMENTS = {
    "baseline": dict(
        video=DATA_DIR / "2026.4.15-pyrrole-baseline.mov",
        crop=dict(y0=790, y1=884, x0=1175, x1=1415),
        n_digits=4, v_min=6.5, v_max=9.0,
    ),
    "decell_1soak": dict(
        video=DATA_DIR / "2026.4.15-decell_chicken_1_soak.mov",
        crop=dict(y0=812, y1=909, x0=1095, x1=1372),
        n_digits=5, v_min=9.0, v_max=14.0,
    ),
    "decell_3soak": dict(
        video=DATA_DIR / "2026.4.15-decell_chicken_3_soak.mov",
        crop=dict(y0=692, y1=782, x0=1120, x1=1392),
        n_digits=4, v_min=1.0, v_max=6.0,
    ),
    "decell_fresh": dict(
        video=DATA_DIR / "2026.4.15-decell_chicken_fresh.mov",
        crop=dict(y0=819, y1=907, x0=1010, x1=1285),
        n_digits=4, v_min=1.0, v_max=6.0,
    ),
    "decell_2soak": dict(
        video=DATA_DIR / "2026.4.16-decel-2soakANDelectropolymeization.mov",
        crop=dict(y0=690, y1=790, x0=1175, x1=1450),
        n_digits=4, v_min=6.5, v_max=9.0,
    ),
    "decell_3soak_dry_rewet": dict(
        video=DATA_DIR / "2026.4.16-decel-3soak-dry-then-rewet.mov",
        crop=dict(y0=731, y1=825, x0=1114, x1=1354),
        n_digits=4, v_min=8.0, v_max=11.0,
    ),
}


def assemble(d):
    i = d[:-DECIMAL_FROM_RIGHT] or "0"
    return float(f"{i}.{d[-DECIMAL_FROM_RIGHT:]}")


def ocr_label(reader, crop_bgr, n_digits, v_min, v_max):
    res = reader.readtext(crop_bgr, detail=1, allowlist="0123456789.")
    if not res:
        return None
    best = max(res, key=lambda r: r[2])
    s = best[1].replace(".", "").strip()
    if not s.isdigit():
        return None
    if len(s) == n_digits and v_min <= assemble(s) <= v_max:
        return s
    if len(s) == n_digits - 1:
        for pref in ("0", "1"):
            c = pref + s
            if v_min <= assemble(c) <= v_max:
                return c
    if len(s) == n_digits + 1 and v_min <= assemble(s[1:]) <= v_max:
        return s[1:]
    return None


def existing_indices(tag):
    idxs = set()
    for p in LABEL_DIR.glob(f"*_{tag}_*.png"):
        stem = p.stem
        try:
            idxs.add(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return idxs


def harvest(reader, tag, cfg, n):
    cap = cv2.VideoCapture(str(cfg["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    used = existing_indices(tag)
    next_idx = (max(used) + 1) if used else 0
    times = np.linspace(0.02 * dur, 0.98 * dur, n)
    saved = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        c = cfg["crop"]
        crop = frame[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        lbl = ocr_label(reader, crop, cfg["n_digits"], cfg["v_min"], cfg["v_max"])
        if lbl is None:
            lbl = "X" * cfg["n_digits"]
        fname = LABEL_DIR / f"{lbl}_{tag}_{next_idx:03d}.png"
        cv2.imwrite(str(fname), crop)
        next_idx += 1
        saved += 1
    cap.release()
    print(f"[{tag}] saved {saved} crops → {LABEL_DIR}")


if __name__ == "__main__":
    print("Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    for tag, cfg in EXPERIMENTS.items():
        harvest(reader, tag, cfg, N_PER_VIDEO)
    total = len(list(LABEL_DIR.glob("*.png")))
    print(f"\nTotal crops now: {total}")
