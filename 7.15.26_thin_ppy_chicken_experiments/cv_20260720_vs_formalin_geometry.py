"""
Where the 2026-07-20 1000um slice lands against the 1/2/3/4-day formalin-fixed samples.

The formalin set (folder 6.16.26_formalin_fixed_decell_chicken_experiments) found
cross-section AREA to be the dominant correlate of actuation, stronger than fixation
day. That set spans only 1.26-1.74 mm^2. The 07.20 sample measures 1.08 mm^2, so it
extends the tissue area range downward by ~15% and acts as an out-of-sample test of
the area trend rather than another point inside it.

All values are final-cycle swings. NOTE the correction pipelines differ slightly:
the formalin notebooks use a global linear-slippage fit across all three scans, this
07.20 run uses per-cycle creep removal.
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
NEW = pd.DataFrame([
    dict(label="07.20 1000um", day=np.nan, L0=3.2, W=4.0, T=0.27,
         force_mN=12.7911, disp_mm=0.0259),
])

for df in (FORMALIN, NEW):
    df["area_mm2"] = df["W"] * df["T"]
    df["stress_MPa"] = df["force_mN"] / df["area_mm2"] / 1000.0
    df["strain_pct"] = df["disp_mm"] / df["L0"] * 100.0

cols = ["label", "L0", "W", "T", "area_mm2", "force_mN", "disp_mm", "stress_MPa", "strain_pct"]
allx = pd.concat([FORMALIN, NEW], ignore_index=True)
pd.set_option("display.width", 200, "display.max_columns", 20)
print("=== all samples ===")
print(allx[cols].round(4).to_string(index=False))

print("\n=== area correlations ===")
for resp in ["strain_pct", "stress_MPa", "force_mN", "disp_mm"]:
    r_old = stats.pearsonr(FORMALIN["area_mm2"], FORMALIN[resp])
    r_new = stats.pearsonr(allx["area_mm2"], allx[resp])
    print(f"  {resp:11s} vs area:  formalin-only r={r_old[0]:+.3f} (p={r_old[1]:.3f}, n=5)"
          f"   with 07.20 r={r_new[0]:+.3f} (p={r_new[1]:.3f}, n=6)")

# out-of-sample check: fit on formalin only, predict the new point
print("\n=== out-of-sample prediction (fit on the 5 formalin samples, predict 07.20) ===")
for resp, unit in [("strain_pct", "%"), ("stress_MPa", "MPa")]:
    sl, ic, r, p, se = stats.linregress(FORMALIN["area_mm2"], FORMALIN[resp])
    pred = sl * NEW["area_mm2"].iloc[0] + ic
    obs = NEW[resp].iloc[0]
    print(f"  {resp:11s}: predicted {pred:.4f} {unit}, observed {obs:.4f} {unit} "
          f"({obs/pred:.2f}x predicted)")

# ============================== FIGURE ======================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

for ax, resp, ylab in [(axes[0], "strain_pct", "Strain swing (%)"),
                       (axes[1], "stress_MPa", "Stress swing (MPa)")]:
    ax.scatter(FORMALIN["area_mm2"], FORMALIN[resp], s=95, color="#2b6cb0",
               ec="k", zorder=4, label="formalin-fixed (1-4 day)")
    ax.scatter(NEW["area_mm2"], NEW[resp], s=190, color="#e53e3e", marker="*",
               ec="k", zorder=5, label="07.20 1000um slice")
    for _, r in pd.concat([FORMALIN, NEW]).iterrows():
        ax.annotate(r["label"], (r["area_mm2"], r[resp]), textcoords="offset points",
                    xytext=(7, 6), fontsize=8)
    sl, ic, rr, pp, se = stats.linregress(FORMALIN["area_mm2"], FORMALIN[resp])
    xs = np.linspace(1.0, 1.85, 50)
    ax.plot(xs, sl * xs + ic, ls="--", color="#2b6cb0", lw=1.6, alpha=0.8,
            label=f"fit on formalin only (r={rr:+.2f})")
    ax.set_xlabel("cross-section area (mm$^2$)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("07.20 slice vs the formalin-fixed set - actuation against cross-section area",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("cv_20260720_vs_formalin_geometry.png", bbox_inches="tight")
print("\nsaved cv_20260720_vs_formalin_geometry.png")
