import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

CV_DIR = os.path.join(os.path.dirname(__file__), "cyclic_voltametry_curves")
files = sorted(glob.glob(os.path.join(CV_DIR, "*.csv")))


def load_cv(path):
    # UTF-16 encoded with header rows; data starts after the "V, uA" line
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
    if data_start is None:
        raise ValueError(f"Could not find data header in {path}")
    rows = []
    for ln in lines[data_start:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        rows.append(parts)
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    # pairs of (V, I) per cycle
    n_cycles = df.shape[1] // 2
    cycles = []
    for k in range(n_cycles):
        v = df.iloc[:, 2 * k].to_numpy()
        i = df.iloc[:, 2 * k + 1].to_numpy()
        mask = ~(pd.isna(v) | pd.isna(i))
        cycles.append((v[mask], i[mask]))
    return cycles


import math
ncols = 3
nrows = math.ceil(len(files) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.ravel()

for ax, path in zip(axes, files):
    cycles = load_cv(path)
    cmap = plt.get_cmap("viridis")
    n = len(cycles)
    for k, (v, i) in enumerate(cycles):
        color = cmap(k / max(n - 1, 1))
        ax.plot(v, i, color=color, lw=1.3, label=f"Cycle {k + 1}")
    name = os.path.basename(path).replace(".csv", "")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current (µA)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

for ax in axes[len(files):]:
    ax.set_visible(False)

fig.suptitle("Cyclic Voltammetry — decell chicken constructs (2026-04-15)", fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "cyclic_voltametry_curves.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
