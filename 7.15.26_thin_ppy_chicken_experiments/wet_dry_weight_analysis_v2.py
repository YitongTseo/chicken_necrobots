"""
Wet -> dry weight modeling for PPy chickenbot samples (07.20.26 dataset).

Because a sample cannot be dried and then re-wet, sacrificial samples are dried
at each stage to build a wet->dry model. Three stages have calibration pairs:

  1. Decellularized:  14 sacrificial pairs (6 from 07.15.26 + 8 new)
  2. Chem Pol Round 1: 5 sacrificial pairs
  3. Chem Pol Round 2: 5 sacrificial pairs (1 physically impossible -> dropped)

All fits are forced through the origin (dry = m*wet, no intercept): a sample with
zero wet mass must have zero dry mass, so a nonzero intercept is unphysical.
"""

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# the exported sheet name drifts between "pyrrole" and "pyyrole", so match loosely
CSV = sorted(glob.glob("07.20.26-py*role chickenbots weights*.csv"))[-1]
print(f"reading {CSV}")
df = pd.read_csv(CSV)
df.columns = [
    "type",
    "decell_wet",
    "decell_dry",
    "cp1_wet",
    "cp1_dry",
    "electropol_wet",
    "cp2_wet",
    "cp2_dry",
]
# the electropol column holds a placeholder string, not data
df["electropol_wet"] = pd.to_numeric(df["electropol_wet"], errors="coerce")

# ---- outlier screening -----------------------------------------------------
# A dry weight can never exceed its own wet weight, and a stage-to-stage wet
# weight cannot jump by an order of magnitude. Both flagged rows look like
# decimal-entry slips (0.0620 vs 0.0062; 0.435 vs 0.0435).
bad_dry = df["cp2_dry"] > df["cp2_wet"]
bad_wet = df["cp2_wet"] > 3 * df["cp1_wet"]

if bad_dry.any():
    print("DROPPED (dry > wet, impossible):")
    print(df.loc[bad_dry, ["type", "cp1_wet", "cp2_wet", "cp2_dry"]].to_string(index=False))
    df.loc[bad_dry, "cp2_dry"] = np.nan
if bad_wet.any():
    print("\nDROPPED (wet weight jumped >3x vs Chem Pol 1, impossible):")
    print(df.loc[bad_wet, ["type", "decell_wet", "cp1_wet", "cp2_wet"]].to_string(index=False))
    df.loc[bad_wet, "cp2_wet"] = np.nan

# ---- calibration pairs -----------------------------------------------------
decell_cal = df.dropna(subset=["decell_wet", "decell_dry"])
cp1_cal = df.dropna(subset=["cp1_wet", "cp1_dry"])
cp2_cal = df.dropna(subset=["cp2_wet", "cp2_dry"])


class OriginFit:
    """Least-squares fit of dry = slope * wet, constrained through (0,0)."""

    def __init__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        self.x, self.y, self.n = x, y, len(x)
        self.slope = np.sum(x * y) / np.sum(x * x)
        resid = y - self.slope * x
        self.ss_res = np.sum(resid**2)
        # 1 free parameter -> n-1 residual dof
        self.s_err = np.sqrt(self.ss_res / (self.n - 1))
        self.stderr = self.s_err / np.sqrt(np.sum(x * x))
        # R^2 measured against the data mean, so it is comparable to an
        # ordinary regression's R^2 (can go negative if the fit is poor)
        self.r2 = 1 - self.ss_res / np.sum((y - y.mean()) ** 2)
        tval = stats.t.ppf(0.975, self.n - 1)
        self.ci = (self.slope - tval * self.stderr, self.slope + tval * self.stderr)

    def predict(self, x):
        return self.slope * np.asarray(x, float)

    def band(self, xs):
        """95% confidence band on the fitted line."""
        tval = stats.t.ppf(0.975, self.n - 1)
        se = self.s_err * np.abs(xs) / np.sqrt(np.sum(self.x**2))
        return self.predict(xs) - tval * se, self.predict(xs) + tval * se


FITS = {
    "Decellularized": OriginFit(decell_cal["decell_wet"], decell_cal["decell_dry"]),
    "Chem Pol Round 1": OriginFit(cp1_cal["cp1_wet"], cp1_cal["cp1_dry"]),
    "Chem Pol Round 2": OriginFit(cp2_cal["cp2_wet"], cp2_cal["cp2_dry"]),
}

print("\n" + "=" * 62)
print("WET -> DRY FITS (forced through origin)")
print("=" * 62)
for name, f in FITS.items():
    print(f"\n{name}:  n = {f.n}")
    print(f"  dry = {f.slope:.4f} * wet")
    print(f"  dry fraction = {f.slope*100:.2f}%  (95% CI {f.ci[0]*100:.2f}-{f.ci[1]*100:.2f}%)")
    print(f"  R^2 (vs mean) = {f.r2:.4f}   residual SD = {f.s_err*1000:.3f} mg")

# ---- apply models to every real sample --------------------------------------
samples = df[df["cp1_wet"].notna()].copy()

GROUP_MAP = {
    "PERPINDICULAR TO GRAIN (500um)": "500um (along + perp)",
    "ALONG THE GRAIN (500um)": "500um (along + perp)",
}
samples["group"] = samples["type"].replace(GROUP_MAP)

# dry weight at each stage: measured where available, else model estimate
samples["decell_dry_est"] = FITS["Decellularized"].predict(samples["decell_wet"])
samples["cp1_dry_est"] = samples["cp1_dry"].fillna(
    pd.Series(FITS["Chem Pol Round 1"].predict(samples["cp1_wet"]), index=samples.index)
)
samples["cp2_dry_est"] = samples["cp2_dry"].fillna(
    pd.Series(FITS["Chem Pol Round 2"].predict(samples["cp2_wet"]), index=samples.index)
)

# changes relative to the decellularized starting point
for stage in ["cp1", "cp2"]:
    samples[f"{stage}_wet_change_pct"] = (
        100 * (samples[f"{stage}_wet"] - samples["decell_wet"]) / samples["decell_wet"]
    )
    samples[f"{stage}_dry_change_pct"] = (
        100 * (samples[f"{stage}_dry_est"] - samples["decell_dry_est"]) / samples["decell_dry_est"]
    )

samples["wet_change"] = samples["cp1_wet"] - samples["decell_wet"]
samples["dry_change"] = samples["cp1_dry_est"] - samples["decell_dry_est"]

out_cols = [
    "type", "group", "decell_wet", "decell_dry_est",
    "cp1_wet", "cp1_dry_est", "cp2_wet", "cp2_dry_est",
    "cp1_wet_change_pct", "cp2_wet_change_pct",
    "cp1_dry_change_pct", "cp2_dry_change_pct",
]
samples[out_cols].round(5).to_csv("weight_change_results_v2.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 25)
print("\n=== Per-sample results ===")
print(samples[out_cols].round(4).to_string(index=False))

print("\n=== Group means (% change from decellularized) ===")
print(
    samples.groupby("group")[
        ["cp1_wet_change_pct", "cp2_wet_change_pct",
         "cp1_dry_change_pct", "cp2_dry_change_pct"]
    ].agg(["mean", "std", "count"]).round(1).to_string()
)

# ============================ VISUALS =======================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# --- 1. calibration fits -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
panels = [
    ("Decellularized", decell_cal, "decell_wet", "decell_dry"),
    ("Chem Pol Round 1", cp1_cal, "cp1_wet", "cp1_dry"),
    ("Chem Pol Round 2", cp2_cal, "cp2_wet", "cp2_dry"),
]
for ax, (name, cal, wcol, dcol) in zip(axes, panels):
    f = FITS[name]
    ax.scatter(cal[wcol], cal[dcol], s=60, color="#2b6cb0", zorder=3,
               label="sacrificial pairs")
    xs = np.linspace(0, cal[wcol].max() * 1.08, 100)
    ax.plot(xs, f.predict(xs), color="#e53e3e", lw=2,
            label=f"dry = {f.slope:.3f}·wet")
    lo, hi = f.band(xs)
    ax.fill_between(xs, lo, hi, color="#e53e3e", alpha=0.15, label="95% CI")
    ax.set_xlabel(f"{name} wet weight (g)")
    ax.set_ylabel(f"{name} dry weight (g)")
    ax.set_title(f"{name}\ndry fraction = {f.slope*100:.1f}%,  R² = {f.r2:.3f},  n = {f.n}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.grid(alpha=0.3)
fig.suptitle("Wet → Dry calibration, forced through origin", fontweight="bold")
fig.tight_layout()
fig.savefig("wet_dry_fits_v2.png", bbox_inches="tight")
print("\nsaved wet_dry_fits_v2.png")

# --- 2. trajectories over stages --------------------------------------------
STAGES = ["Decell", "Chem Pol 1", "Chem Pol 2"]
ORDER = [
    "AGAINST THE GRAIN (1000um)",
    "ALONG THE GRAIN (1000um)",
    "500um (along + perp)",
]
colors = dict(zip(ORDER, plt.cm.viridis(np.linspace(0.15, 0.75, len(ORDER)))))


def trajectory_figure(cols, ylabel, title, fname, unit_scale=1000):
    """Left: absolute mass per stage. Right: % change vs decell.

    Both panels use the same idiom: every sample as a jittered dot, groups dodged
    sideways, mean +/- 1 SD drawn over the cloud. Absolute weight is on the left
    so spread in milligrams is legible directly, rather than only as a ratio.

    Every sample is plotted at every stage it has data for, so all 24 decell and
    24 Chem Pol 1 measurements appear even though only 18 reached Chem Pol 2.
    That means the solid mean line averages a different roster at each stage, so
    the complete-case mean (samples present at ALL stages) is overlaid dashed.
    Where the two lines diverge, the movement is survivorship, not chemistry.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    xs = np.arange(len(STAGES))
    complete = samples.dropna(subset=cols)
    base = "wet" if "wet" in cols[1] else "dry"

    # Groups are dodged sideways and points jittered within each dodge, so with
    # n as low as 4 the reader can see the actual spread rather than trusting an
    # SD whisker drawn from a handful of samples.
    dodge = np.linspace(-0.13, 0.13, len(ORDER))
    rng = np.random.default_rng(0)  # seeded: jitter is stable across re-runs

    # --- left: absolute mass, same dot-cloud treatment as the % panel ---------
    for g, dx in zip(ORDER, dodge):
        sub = samples[samples["group"] == g]
        csub = complete[complete["group"] == g]
        means = [sub[c].mean() * unit_scale for c in cols]
        sds = [sub[c].std() * unit_scale for c in cols]
        ns = [int(sub[c].notna().sum()) for c in cols]

        for xi, c in zip(xs, cols):
            vals = sub[c].dropna().values * unit_scale
            jit = rng.uniform(-0.05, 0.05, len(vals))
            ax1.scatter(np.full(len(vals), xi + dx) + jit, vals, s=44,
                        color=colors[g], alpha=0.5, edgecolor="white",
                        linewidth=0.6, zorder=3)

        ax1.plot(xs + dx, [csub[c].mean() * unit_scale for c in cols], "--",
                 color=colors[g], lw=1.6, alpha=0.75, zorder=4)
        ax1.errorbar(xs + dx, means, yerr=sds, fmt="-o", color=colors[g], lw=2.5,
                     ms=9, capsize=4, elinewidth=1.4, markeredgecolor="k",
                     zorder=5, label=f"{g}  (n={ns[0]}/{ns[1]}/{ns[2]})")
    ax1.plot([], [], "--", color="0.4", lw=1.6, label="complete cases only")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(STAGES)
    ax1.set_ylabel(ylabel)
    ax1.set_title("Absolute weight (bars = 1 SD, dots = individual samples)")
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0)

    for gi, (g, dx) in enumerate(zip(ORDER, dodge)):
        sub = samples[samples["group"] == g]
        csub = complete[complete["group"] == g]
        pcols = [f"{s}_{base}_change_pct" for s in ["cp1", "cp2"]]
        pc = [0] + [sub[c].mean() for c in pcols]
        sd = [0] + [sub[c].std() for c in pcols]
        ns = [len(sub)] + [int(sub[c].notna().sum()) for c in pcols]

        # skip the Decell column: every sample is 0% there by definition
        for xi, c in zip(xs[1:], pcols):
            vals = sub[c].dropna().values
            jit = rng.uniform(-0.05, 0.05, len(vals))
            ax2.scatter(np.full(len(vals), xi + dx) + jit, vals, s=44,
                        color=colors[g], alpha=0.5, edgecolor="white",
                        linewidth=0.6, zorder=3)

        ax2.plot(xs + dx, [0] + [csub[c].mean() for c in pcols], "--",
                 color=colors[g], lw=1.6, alpha=0.75, zorder=4)
        ax2.errorbar(xs + dx, pc, yerr=sd, fmt="-o", color=colors[g], lw=2.5,
                     ms=9, capsize=4, elinewidth=1.4, markeredgecolor="k",
                     zorder=5, label=f"{g}  (n={ns[0]}/{ns[1]}/{ns[2]})")
        # stagger label height by group so the three dodged means don't collide
        dy = [12, -2, -16][gi]
        for xi, val in zip(xs[1:], pc[1:]):
            ax2.annotate(f"{val:+.0f}%", (xi + dx, val),
                         textcoords="offset points", xytext=(12, dy), fontsize=9,
                         fontweight="bold", color=colors[g], zorder=6)
    ax2.plot([], [], "--", color="0.4", lw=1.6, label="complete cases only")
    ax2.axhline(0, color="k", lw=0.9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(STAGES)
    ax2.set_ylabel("Change from decellularized (%)")
    ax2.set_title("Mean % change (bars = 1 SD, dots = individual samples)")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    print(f"saved {fname}")


trajectory_figure(
    ["decell_wet", "cp1_wet", "cp2_wet"],
    "Wet weight (mg)",
    "WET weight across polymerization stages (measured)",
    "wet_weight_over_time_v2.png",
)
trajectory_figure(
    ["decell_dry_est", "cp1_dry_est", "cp2_dry_est"],
    "Dry weight (mg)",
    "DRY weight across polymerization stages (model-estimated)",
    "dry_weight_over_time_v2.png",
)
