"""
Where the 2026-07-29 "ALONG" slice (new electropol cell) lands against every previously
measured sample: the 1/2/3/4-day formalin-fixed set and the 2026-07-20 1000 um slice.

Extends 7.15.26_thin_ppy_chicken_experiments/cv_20260720_vs_formalin_geometry.py with the
07.29 point, now that its geometry is measured (W 5.0 mm, L0 4.0 mm, T 0.2230 mm).

All values are final-cycle (cycle 3) swings. NOTE the correction pipelines differ slightly:
the formalin notebooks use a global linear-slippage fit across all three scans, while the
07.20 and 07.29 runs use per-cycle creep removal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# formalin set: values read from geometry_correlation.ipynb (executed outputs)
FORMALIN = pd.DataFrame([
    dict(label="1-day",    day=1, L0=7.0,  W=4.0, T=0.360, force_mN=13.8403, disp_mm=0.0247),
    dict(label="2-day",    day=2, L0=9.0,  W=2.8, T=0.450, force_mN=23.5770, disp_mm=0.0477),
    dict(label="3-day t1", day=3, L0=5.0,  W=3.1, T=0.426, force_mN=11.8312, disp_mm=0.0331),
    dict(label="3-day t2", day=3, L0=11.0, W=2.5, T=0.696, force_mN=8.0013,  disp_mm=0.0122),
    dict(label="4-day",    day=4, L0=11.0, W=3.4, T=0.374, force_mN=29.1361, disp_mm=0.0780),
])
PRIOR = pd.DataFrame([
    dict(label="07.20 1000um", L0=3.2, W=4.0, T=0.270, force_mN=12.7911, disp_mm=0.0259),
])
NEW = pd.DataFrame([
    dict(label="07.29 ALONG (new cell)", L0=4.0, W=5.0, T=0.2230,
         force_mN=22.8683, disp_mm=0.0524),
])

for df in (FORMALIN, PRIOR, NEW):
    df["area_mm2"] = df["W"] * df["T"]
    df["stress_MPa"] = df["force_mN"] / df["area_mm2"] / 1000.0
    df["strain_pct"] = df["disp_mm"] / df["L0"] * 100.0

allx = pd.concat([FORMALIN, PRIOR, NEW], ignore_index=True)
prev = pd.concat([FORMALIN, PRIOR], ignore_index=True)
cols = ["label", "L0", "W", "T", "area_mm2", "force_mN", "disp_mm", "stress_MPa",
        "strain_pct"]
pd.set_option("display.width", 220, "display.max_columns", 20)
print("=== all samples (final-cycle swings) ===")
print(allx[cols].round(4).to_string(index=False))

print("\n=== where 07.29 ranks (1 = highest) ===")
for m in ["force_mN", "disp_mm", "stress_MPa", "strain_pct"]:
    rank = int((allx[m] > NEW[m].iloc[0]).sum()) + 1
    print(f"  {m:11s}: {NEW[m].iloc[0]:.4f}  -> rank {rank} of {len(allx)}"
          f"   (previous best {prev[m].max():.4f}, median {prev[m].median():.4f})")

print("\n=== area correlations ===")
for resp in ["strain_pct", "stress_MPa", "force_mN", "disp_mm"]:
    a = stats.pearsonr(FORMALIN["area_mm2"], FORMALIN[resp])
    b = stats.pearsonr(allx["area_mm2"], allx[resp])
    print(f"  {resp:11s} vs area:  formalin-only r={a[0]:+.3f} (p={a[1]:.3f}, n=5)"
          f"   all samples r={b[0]:+.3f} (p={b[1]:.3f}, n={len(allx)})")

print("\n=== out-of-sample check: fit on the 5 formalin samples, predict 07.29 ===")
for resp, unit in [("strain_pct", "%"), ("stress_MPa", "MPa")]:
    sl, ic, r, p, se = stats.linregress(FORMALIN["area_mm2"], FORMALIN[resp])
    pred = sl * NEW["area_mm2"].iloc[0] + ic
    obs = NEW[resp].iloc[0]
    print(f"  {resp:11s}: predicted {pred:.4f} {unit}, observed {obs:.4f} {unit} "
          f"({obs/pred:.2f}x predicted)")

# ============================== FIGURE ======================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

PANELS = [
    (axes[0, 0], "area_mm2", "strain_pct", "cross-section area (mm$^2$)",
     "Strain swing (%)", True),
    (axes[0, 1], "area_mm2", "stress_MPa", "cross-section area (mm$^2$)",
     "Stress swing (MPa)", True),
    (axes[1, 0], "area_mm2", "force_mN", "cross-section area (mm$^2$)",
     "Force swing (mN)", False),
    (axes[1, 1], "L0", "disp_mm", "gauge length L0 (mm)",
     "Displacement swing (mm)", False),
]

for ax, xk, yk, xlab, ylab, fitline in PANELS:
    ax.scatter(FORMALIN[xk], FORMALIN[yk], s=95, color="#2b6cb0", ec="k", zorder=4,
               label="formalin-fixed (1-4 day)")
    ax.scatter(PRIOR[xk], PRIOR[yk], s=170, color="#dd6b20", marker="D", ec="k",
               zorder=5, label="07.20 1000um")
    ax.scatter(NEW[xk], NEW[yk], s=320, color="#e53e3e", marker="*", ec="k", zorder=6,
               label="07.29 ALONG (new cell)")
    for _, r in allx.iterrows():
        ax.annotate(r["label"], (r[xk], r[yk]), textcoords="offset points",
                    xytext=(8, 6), fontsize=7.5)
    if fitline:
        sl, ic, rr, pp, se = stats.linregress(FORMALIN[xk], FORMALIN[yk])
        xs = np.linspace(allx[xk].min() * 0.95, allx[xk].max() * 1.05, 50)
        ax.plot(xs, sl * xs + ic, ls="--", color="#2b6cb0", lw=1.6, alpha=0.8,
                label=f"fit on formalin only (r={rr:+.2f})")
    ax.set(xlabel=xlab, ylabel=ylab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

fig.suptitle("2026-07-29 ALONG slice (new electropol cell) vs all previously measured "
             "samples\nfinal-cycle swings; geometry measured for every point",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("cv_20260729_vs_all_samples.png", bbox_inches="tight")
print("\nsaved cv_20260729_vs_all_samples.png")
