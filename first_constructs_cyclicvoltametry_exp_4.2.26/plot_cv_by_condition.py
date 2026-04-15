import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CV_DIR = os.path.join(HERE, "cyclic_voltametry_exps")
CYCLE_INDEX = 2  # 0-indexed → 3rd cycle


def load_cv(path):
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-16")
    lines = text.splitlines()
    data_start = None
    for i, ln in enumerate(lines):
        parts = [p.strip() for p in ln.split(",")]
        if parts and parts[0] == "V" and len(parts) >= 2:
            data_start = i + 1
            break
    rows = [ln.split(",") for ln in lines[data_start:] if ln.strip()]
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    n_cycles = df.shape[1] // 2
    cycles = []
    for k in range(n_cycles):
        v = df.iloc[:, 2 * k].to_numpy()
        i = df.iloc[:, 2 * k + 1].to_numpy()
        mask = ~(pd.isna(v) | pd.isna(i))
        cycles.append((v[mask], i[mask]))
    return cycles


def condition_of(fname):
    if "0cycle_nopyrroleDBS" in fname:
        return "0cycle (no pyrroleDBS)"
    if "1cycle_pyrroleDBS" in fname:
        return "1cycle (pyrroleDBS)"
    return "pyrroleDBS (baseline)"


def scan_rate_of(fname):
    m = re.search(r"(\d+)mVperSec", fname)
    return int(m.group(1))


files = sorted(glob.glob(os.path.join(CV_DIR, "*.csv")))
by_cond = {}
for p in files:
    fname = os.path.basename(p)
    by_cond.setdefault(condition_of(fname), []).append(p)

cond_order = ["pyrroleDBS (baseline)", "0cycle (no pyrroleDBS)", "1cycle (pyrroleDBS)"]
rates_all = sorted({scan_rate_of(os.path.basename(p)) for p in files})
cmap = plt.get_cmap("plasma")
rate_colors = {r: cmap(1 - i / max(len(rates_all) - 1, 1)) for i, r in enumerate(rates_all)}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True)

for ax, cond in zip(axes, cond_order):
    paths = sorted(by_cond.get(cond, []), key=lambda p: scan_rate_of(os.path.basename(p)))
    for p in paths:
        rate = scan_rate_of(os.path.basename(p))
        cycles = load_cv(p)
        idx = CYCLE_INDEX if len(cycles) > CYCLE_INDEX else len(cycles) - 1
        v, i = cycles[idx]
        ax.plot(v, i, color=rate_colors[rate], lw=1.4,
                label=f"{rate} mV/s (cycle {idx + 1})")
    ax.set_title(cond, fontsize=11)
    ax.set_xlabel("Potential (V)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

axes[0].set_ylabel("Current (µA)")
fig.suptitle(f"Cyclic Voltammetry — cycle {CYCLE_INDEX + 1} overlay across scan rates (2026-03-31)",
             fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "cv_by_condition_cycle3.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
