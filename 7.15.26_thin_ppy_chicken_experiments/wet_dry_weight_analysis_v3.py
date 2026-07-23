"""
Wet -> dry weight modeling for PPy chickenbot samples (07.23.26 dataset).

This is the "dunking" chem-pol method, so there are only TWO stages instead of
the 07.20.26 electropolymerization method's three:

  1. Decellularized:        14 sacrificial wet/dry pairs (same as 07.20 sheet)
  2. 4 Dunks - Chem Pol:     wet/dry pairs come from the run samples that happened
                             to also get a dry weight taken

As before every fit is forced through the origin (dry = m*wet, no intercept):
a sample with zero wet mass must have zero dry mass.
"""

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV = sorted(glob.glob("07.23*chickenbots weights*dunking*.csv"))[-1]
print(f"reading {CSV}")
df = pd.read_csv(CSV)
df.columns = ["type", "decell_wet", "decell_dry", "cp_wet", "cp_dry"]
for c in ["decell_wet", "decell_dry", "cp_wet", "cp_dry"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- outlier screening -----------------------------------------------------
# Physical checks, same spirit as v2:
#   * a dry weight can never exceed its own wet weight
#   * the dunk chem-pol step REDUCES wet mass, so cp_wet > decell_wet is impossible
#   * decell_wet cannot be many times larger than the same sample's cp_wet
# Every flag here is an exact 10x decimal slip (value/10 restores physical
# plausibility), so we correct by /10 and print each change loudly rather than
# silently dropping the sample -- the corrected number lands in the normal range.
def fix_decimal_slip(mask, col, reason):
    for idx in df.index[mask]:
        old = df.at[idx, col]
        new = old / 10.0
        print(f"  CORRECTED row {idx}: {col} {old:g} -> {new:g}   ({reason})")
        df.at[idx, col] = new

print("\nDecimal-slip screening (10x entry errors):")
fix_decimal_slip(df["cp_wet"] > df["decell_wet"], "cp_wet",
                 "chem-pol wet exceeded decell wet")
fix_decimal_slip(df["decell_wet"] > 4 * df["cp_wet"], "decell_wet",
                 "decell wet was >4x its chem-pol wet")
# any residual dry>wet after the above is genuinely unusable -> drop that cell
bad_dry = df["cp_dry"] > df["cp_wet"]
if bad_dry.any():
    print("  DROPPED (dry > wet, unrecoverable):")
    print(df.loc[bad_dry, ["type", "cp_wet", "cp_dry"]].to_string())
    df.loc[bad_dry, "cp_dry"] = np.nan

# ---- calibration pairs -----------------------------------------------------
decell_cal = df.dropna(subset=["decell_wet", "decell_dry"])
cp_cal = df.dropna(subset=["cp_wet", "cp_dry"])


class OriginFit:
    """Least-squares fit of dry = slope * wet, constrained through (0,0)."""

    def __init__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        self.x, self.y, self.n = x, y, len(x)
        self.slope = np.sum(x * y) / np.sum(x * x)
        resid = y - self.slope * x
        self.ss_res = np.sum(resid**2)
        self.s_err = np.sqrt(self.ss_res / (self.n - 1))
        self.stderr = self.s_err / np.sqrt(np.sum(x * x))
        self.r2 = 1 - self.ss_res / np.sum((y - y.mean()) ** 2)
        tval = stats.t.ppf(0.975, self.n - 1)
        self.ci = (self.slope - tval * self.stderr, self.slope + tval * self.stderr)

    def predict(self, x):
        return self.slope * np.asarray(x, float)

    def band(self, xs):
        tval = stats.t.ppf(0.975, self.n - 1)
        se = self.s_err * np.abs(xs) / np.sqrt(np.sum(self.x**2))
        return self.predict(xs) - tval * se, self.predict(xs) + tval * se


FITS = {
    "Decellularized": OriginFit(decell_cal["decell_wet"], decell_cal["decell_dry"]),
    "4 Dunks - Chem Pol": OriginFit(cp_cal["cp_wet"], cp_cal["cp_dry"]),
}

print("\n" + "=" * 62)
print("WET -> DRY FITS (forced through origin)")
print("=" * 62)
for name, f in FITS.items():
    print(f"\n{name}:  n = {f.n}")
    print(f"  dry = {f.slope:.4f} * wet")
    print(f"  dry fraction = {f.slope*100:.2f}%  (95% CI {f.ci[0]*100:.2f}-{f.ci[1]*100:.2f}%)")
    print(f"  R^2 (vs mean) = {f.r2:.4f}   residual SD = {f.s_err*1000:.3f} mg")

# ---- apply models to every real run sample ---------------------------------
samples = df[df["cp_wet"].notna()].copy()

# normalize the free-text type labels into the four grain x thickness groups
def norm_group(t):
    t = str(t).upper()
    grain = "AGAINST" if "AGAINST" in t else ("ALONG" if "ALONG" in t else "?")
    thick = "1000um" if "1000" in t else ("500um" if "500" in t else "?")
    return f"{grain} {thick}"

samples["group"] = samples["type"].apply(norm_group)

samples["decell_dry_est"] = FITS["Decellularized"].predict(samples["decell_wet"])
samples["cp_dry_est"] = samples["cp_dry"].fillna(
    pd.Series(FITS["4 Dunks - Chem Pol"].predict(samples["cp_wet"]), index=samples.index)
)

samples["wet_change_pct"] = 100 * (samples["cp_wet"] - samples["decell_wet"]) / samples["decell_wet"]
samples["dry_change_pct"] = 100 * (samples["cp_dry_est"] - samples["decell_dry_est"]) / samples["decell_dry_est"]

out_cols = [
    "type", "group", "decell_wet", "decell_dry_est",
    "cp_wet", "cp_dry_est", "wet_change_pct", "dry_change_pct",
]
samples[out_cols].round(5).to_csv("weight_change_results_v3.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 25)
print("\n=== Per-sample results ===")
print(samples[out_cols].round(4).to_string(index=False))

print("\n=== Group means (% change from decellularized) ===")
print(
    samples.groupby("group")[["wet_change_pct", "dry_change_pct"]]
    .agg(["mean", "std", "count"]).round(1).to_string()
)

# ============================ VISUALS =======================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# --- 1. calibration fits -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
panels = [
    ("Decellularized", decell_cal, "decell_wet", "decell_dry"),
    ("4 Dunks - Chem Pol", cp_cal, "cp_wet", "cp_dry"),
]
for ax, (name, cal, wcol, dcol) in zip(axes, panels):
    f = FITS[name]
    ax.scatter(cal[wcol], cal[dcol], s=60, color="#2b6cb0", zorder=3,
               label="sacrificial pairs")
    xs = np.linspace(0, cal[wcol].max() * 1.08, 100)
    ax.plot(xs, f.predict(xs), color="#e53e3e", lw=2, label=f"dry = {f.slope:.3f}·wet")
    lo, hi = f.band(xs)
    ax.fill_between(xs, lo, hi, color="#e53e3e", alpha=0.15, label="95% CI")
    ax.set_xlabel(f"{name} wet weight (g)")
    ax.set_ylabel(f"{name} dry weight (g)")
    ax.set_title(f"{name}\ndry fraction = {f.slope*100:.1f}%,  R² = {f.r2:.3f},  n = {f.n}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.grid(alpha=0.3)
fig.suptitle("Wet → Dry calibration, forced through origin (dunking method)", fontweight="bold")
fig.tight_layout()
fig.savefig("wet_dry_fits_v3.png", bbox_inches="tight")
print("\nsaved wet_dry_fits_v3.png")

# --- 2. trajectories over the two stages ------------------------------------
STAGES = ["Decell", "4 Dunks\nChem Pol"]
ORDER = ["AGAINST 1000um", "ALONG 1000um", "AGAINST 500um", "ALONG 500um"]
ORDER = [g for g in ORDER if g in samples["group"].unique()]
colors = dict(zip(ORDER, plt.cm.viridis(np.linspace(0.1, 0.8, len(ORDER)))))


def trajectory_figure(cols, ylabel, title, fname, pct_col, unit_scale=1000):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    xs = np.arange(len(STAGES))
    dodge = np.linspace(-0.13, 0.13, len(ORDER))
    rng = np.random.default_rng(0)

    for g, dx in zip(ORDER, dodge):
        sub = samples[samples["group"] == g]
        means = [sub[c].mean() * unit_scale for c in cols]
        sds = [sub[c].std() * unit_scale for c in cols]
        ns = [int(sub[c].notna().sum()) for c in cols]
        for xi, c in zip(xs, cols):
            vals = sub[c].dropna().values * unit_scale
            jit = rng.uniform(-0.05, 0.05, len(vals))
            ax1.scatter(np.full(len(vals), xi + dx) + jit, vals, s=44,
                        color=colors[g], alpha=0.5, edgecolor="white",
                        linewidth=0.6, zorder=3)
        ax1.errorbar(xs + dx, means, yerr=sds, fmt="-o", color=colors[g], lw=2.5,
                     ms=9, capsize=4, elinewidth=1.4, markeredgecolor="k",
                     zorder=5, label=f"{g}  (n={ns[0]}/{ns[1]})")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(STAGES)
    ax1.set_ylabel(ylabel)
    ax1.set_title("Absolute weight (bars = 1 SD, dots = samples)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0)

    for gi, (g, dx) in enumerate(zip(ORDER, dodge)):
        sub = samples[samples["group"] == g]
        pc = [0, sub[pct_col].mean()]
        sd = [0, sub[pct_col].std()]
        n = int(sub[pct_col].notna().sum())
        vals = sub[pct_col].dropna().values
        jit = rng.uniform(-0.05, 0.05, len(vals))
        ax2.scatter(np.full(len(vals), xs[1] + dx) + jit, vals, s=44,
                    color=colors[g], alpha=0.5, edgecolor="white",
                    linewidth=0.6, zorder=3)
        ax2.errorbar(xs + dx, pc, yerr=sd, fmt="-o", color=colors[g], lw=2.5,
                     ms=9, capsize=4, elinewidth=1.4, markeredgecolor="k",
                     zorder=5, label=f"{g}  (n={n})")
        dy = [12, -2, -16, 24][gi % 4]
        ax2.annotate(f"{pc[1]:+.0f}%", (xs[1] + dx, pc[1]),
                     textcoords="offset points", xytext=(12, dy), fontsize=9,
                     fontweight="bold", color=colors[g], zorder=6)
    ax2.axhline(0, color="k", lw=0.9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(STAGES)
    ax2.set_ylabel("Change from decellularized (%)")
    ax2.set_title("Mean % change (bars = 1 SD, dots = samples)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    print(f"saved {fname}")


trajectory_figure(
    ["decell_wet", "cp_wet"], "Wet weight (mg)",
    "WET weight across dunking chem-pol (measured)",
    "wet_weight_over_time_v3.png", "wet_change_pct",
)
trajectory_figure(
    ["decell_dry_est", "cp_dry_est"], "Dry weight (mg)",
    "DRY weight across dunking chem-pol (model-estimated)",
    "dry_weight_over_time_v3.png", "dry_change_pct",
)


# --- 3. statistical outlier flagging ----------------------------------------
# The 10x decimal slips are already handled above; this is a second pass for
# samples whose stage-to-stage BEHAVIOR is anomalous. We use Tukey fences
# (median-robust, no normality assumption) on the two change metrics. A sample
# is an outlier if either its measured wet change or its dry change falls
# outside 1.5*IQR -- wet is directly measured, dry is the model propagation, so
# flagging on either catches both a bad weighing and a runaway estimate.
def tukey_outliers(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series < lo) | (series > hi), (lo, hi)

wet_out, wet_fence = tukey_outliers(samples["wet_change_pct"])
dry_out, dry_fence = tukey_outliers(samples["dry_change_pct"])
samples["is_outlier"] = wet_out | dry_out

print("\n" + "=" * 62)
print("STATISTICAL OUTLIER SCREEN (Tukey 1.5*IQR on change metrics)")
print("=" * 62)
print(f"  wet-change fence: {wet_fence[0]:+.1f}% to {wet_fence[1]:+.1f}%")
print(f"  dry-change fence: {dry_fence[0]:+.1f}% to {dry_fence[1]:+.1f}%")
if samples["is_outlier"].any():
    print("\n  flagged samples:")
    print(samples.loc[samples["is_outlier"],
          ["type", "decell_wet", "cp_wet", "wet_change_pct", "dry_change_pct"]]
          .round(2).to_string())
else:
    print("  none flagged")

clean = samples[~samples["is_outlier"]]


# --- 4. pooled: every sample as ONE group -----------------------------------
def pooled_dry_figure(data, fname, title, outliers=None):
    cols = ["decell_dry_est", "cp_dry_est"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    xs = np.arange(len(STAGES))
    rng = np.random.default_rng(0)
    color = "#2b6cb0"

    def draw(ax, xi, vals, c, marker_alpha=0.45):
        jit = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(np.full(len(vals), xi) + jit, vals, s=48, color=c,
                   alpha=marker_alpha, edgecolor="white", linewidth=0.6, zorder=3)

    means = [data[c].mean() * 1000 for c in cols]
    sds = [data[c].std() * 1000 for c in cols]
    ns = [int(data[c].notna().sum()) for c in cols]
    for xi, c in zip(xs, cols):
        draw(ax1, xi, data[c].dropna().values * 1000, color)
        if outliers is not None and len(outliers):
            draw(ax1, xi, outliers[c].dropna().values * 1000, "#e53e3e", 0.55)
    ax1.errorbar(xs, means, yerr=sds, fmt="-o", color=color, lw=2.5, ms=10,
                 capsize=5, elinewidth=1.5, markeredgecolor="k", zorder=5,
                 label=f"kept samples (n={ns[0]}/{ns[1]})")
    if outliers is not None and len(outliers):
        ax1.scatter([], [], s=48, color="#e53e3e", alpha=0.55,
                    edgecolor="white", label="excluded outlier")
    ax1.set_xticks(xs); ax1.set_xticklabels(STAGES)
    ax1.set_ylabel("Dry weight (mg)")
    ax1.set_title("Absolute dry weight (bars = 1 SD, dots = samples)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_ylim(0)

    pc = [0, data["dry_change_pct"].mean()]
    sd = [0, data["dry_change_pct"].std()]
    n = int(data["dry_change_pct"].notna().sum())
    draw(ax2, xs[1], data["dry_change_pct"].dropna().values, color)
    if outliers is not None and len(outliers):
        draw(ax2, xs[1], outliers["dry_change_pct"].dropna().values, "#e53e3e", 0.55)
    ax2.errorbar(xs, pc, yerr=sd, fmt="-o", color=color, lw=2.5, ms=10,
                 capsize=5, elinewidth=1.5, markeredgecolor="k", zorder=5,
                 label=f"kept samples (n={n})")
    ax2.annotate(f"{pc[1]:+.0f}%", (xs[1], pc[1]), textcoords="offset points",
                 xytext=(12, 8), fontsize=11, fontweight="bold", color=color, zorder=6)
    ax2.axhline(0, color="k", lw=0.9)
    ax2.set_xticks(xs); ax2.set_xticklabels(STAGES)
    ax2.set_ylabel("Change from decellularized (%)")
    ax2.set_title("Mean % change (bars = 1 SD, dots = samples)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    print(f"saved {fname}")


pooled_dry_figure(samples, "dry_weight_over_time_v3_pooled.png",
                  "DRY weight across dunking chem-pol — all groups pooled")
pooled_dry_figure(clean, "dry_weight_over_time_v3_pooled_no_outliers.png",
                  "DRY weight across dunking chem-pol — pooled, outliers removed",
                  outliers=samples[samples["is_outlier"]])


def dry_summary(data, label):
    print(f"\n  [{label}]  n={len(data)}")
    print(f"    Decell dry (mg):   {data['decell_dry_est'].mean()*1000:.2f} +/- "
          f"{data['decell_dry_est'].std()*1000:.2f}")
    print(f"    Chem-pol dry (mg): {data['cp_dry_est'].mean()*1000:.2f} +/- "
          f"{data['cp_dry_est'].std()*1000:.2f}")
    print(f"    Mean dry change:   {data['dry_change_pct'].mean():+.1f}% +/- "
          f"{data['dry_change_pct'].std():.1f}%")

print("\n=== Pooled dry-weight summary ===")
dry_summary(samples, "all samples")
dry_summary(clean, "outliers removed")
