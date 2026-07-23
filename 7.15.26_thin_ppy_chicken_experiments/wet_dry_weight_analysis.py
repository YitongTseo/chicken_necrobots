"""
Wet -> dry weight modeling for PPy chickenbot samples.

Because a sample cannot be dried and then re-wet, sacrificial samples are dried
at each stage to build a linear wet->dry model. Two stages exist here:

  1. Decellularized stage: sacrificial pairs are the unlabeled rows (both wet+dry).
  2. Chem Pol Round 1 stage: sacrificial dry weights recorded on a few real samples.

We fit dry = m*wet + b for each stage, use it to estimate the dry weight of every
sample at every stage, and then compute the dry-mass change across polymerization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV = "pyyrole chickenbots weights - Sheet1.csv"
df = pd.read_csv(CSV)
df.columns = ["type", "decell_wet", "decell_dry", "cp_wet", "cp_dry"]

# ---- calibration pairs -----------------------------------------------------
# Decell model: rows with both decell wet and decell dry
decell_cal = df.dropna(subset=["decell_wet", "decell_dry"])
# Chem-pol model: rows with both chem-pol wet and chem-pol dry
cp_cal = df.dropna(subset=["cp_wet", "cp_dry"])


def fit(x, y):
    res = stats.linregress(x, y)
    return res  # slope, intercept, rvalue, pvalue, stderr, intercept_stderr


decell_fit = fit(decell_cal["decell_wet"].values, decell_cal["decell_dry"].values)
cp_fit = fit(cp_cal["cp_wet"].values, cp_cal["cp_dry"].values)

print("=== Decellularized wet->dry fit ===")
print(f"  n = {len(decell_cal)}")
print(f"  dry = {decell_fit.slope:.4f} * wet + {decell_fit.intercept:+.5f}")
print(f"  R^2 = {decell_fit.rvalue**2:.4f}   p = {decell_fit.pvalue:.4g}")
print(f"  dry-fraction (slope) = {decell_fit.slope*100:.1f}% of wet mass")

print("\n=== Chem Pol Round 1 wet->dry fit ===")
print(f"  n = {len(cp_cal)}")
print(f"  dry = {cp_fit.slope:.4f} * wet + {cp_fit.intercept:+.5f}")
print(f"  R^2 = {cp_fit.rvalue**2:.4f}   p = {cp_fit.pvalue:.4g}")
print(f"  dry-fraction (slope) = {cp_fit.slope*100:.1f}% of wet mass")


def predict(res, x):
    return res.slope * x + res.intercept


# ---- apply models to all real (labeled) samples ----------------------------
samples = df[df["type"].notna()].copy()

# perpendicular-to-grain (500um) and along-the-grain (500um) are effectively the
# same specimen type -> merge into a single 500um group for reporting
GROUP_MAP = {
    "PERPINDICULAR TO GRAIN (500um)": "500um (along + perp)",
    "ALONG THE GRAIN (500um)": "500um (along + perp)",
}
samples["group"] = samples["type"].replace(GROUP_MAP)

samples["decell_dry_est"] = predict(decell_fit, samples["decell_wet"])
# use measured chem-pol dry where present, else model estimate
samples["cp_dry_est"] = predict(cp_fit, samples["cp_wet"])
samples["cp_dry_used"] = samples["cp_dry"].where(
    samples["cp_dry"].notna(), samples["cp_dry_est"]
)

# weight changes across polymerization (dry basis = actual solid material)
samples["dry_change"] = samples["cp_dry_used"] - samples["decell_dry_est"]
samples["dry_change_pct"] = 100 * samples["dry_change"] / samples["decell_dry_est"]
# wet change too (informational)
samples["wet_change"] = samples["cp_wet"] - samples["decell_wet"]
samples["wet_change_pct"] = 100 * samples["wet_change"] / samples["decell_wet"]

out = samples[
    [
        "type",
        "group",
        "decell_wet",
        "decell_dry_est",
        "cp_wet",
        "cp_dry_used",
        "dry_change",
        "dry_change_pct",
        "wet_change",
        "wet_change_pct",
    ]
].copy()
out.to_csv("weight_change_results.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n=== Per-sample weight change ===")
print(out.round(4).to_string(index=False))

print("\n=== Group means (500um groups merged) ===")
print(
    samples.groupby("group")[["wet_change_pct", "dry_change_pct"]]
    .agg(["mean", "std", "count"])
    .round(1)
)

# ============================ VISUALS =======================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

for ax, cal, res, wcol, dcol, title in [
    (axes[0], decell_cal, decell_fit, "decell_wet", "decell_dry", "Decellularized"),
    (axes[1], cp_cal, cp_fit, "cp_wet", "cp_dry", "Chem Pol Round 1"),
]:
    x = cal[wcol].values
    y = cal[dcol].values
    ax.scatter(x, y, s=60, color="#2b6cb0", zorder=3, label="sacrificial pairs")
    xs = np.linspace(0, x.max() * 1.08, 100)
    ax.plot(xs, predict(res, xs), color="#e53e3e", lw=2,
            label=f"dry = {res.slope:.3f}·wet {res.intercept:+.4f}")
    # 95% CI band
    n = len(x)
    tval = stats.t.ppf(0.975, n - 2)
    sy = np.sqrt(np.sum((y - predict(res, x)) ** 2) / (n - 2))
    se = sy * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xs, predict(res, xs) - tval * se, predict(res, xs) + tval * se,
                    color="#e53e3e", alpha=0.15, label="95% CI")
    ax.set_xlabel(f"{title} wet weight (g)")
    ax.set_ylabel(f"{title} dry weight (g)")
    ax.set_title(f"{title}\nR² = {res.rvalue**2:.3f}, n = {n}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.grid(alpha=0.3)

fig.suptitle("Wet → Dry weight calibration models", fontweight="bold")
fig.tight_layout()
fig.savefig("wet_dry_fits.png", bbox_inches="tight")
print("\nsaved wet_dry_fits.png")

# ---- weight-change figures (500um groups merged) --------------------------
# fixed group order, largest specimen first
ORDER = [
    "AGAINST THE GRAIN (1000um)",
    "ALONG THE GRAIN (1000um)",
    "500um (along + perp)",
]
grp = samples.groupby("group")
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ORDER)))


def grouped_bar(metric_col, ylabel, title, fname, absolute_col=None, absunit="g"):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    means = [grp.get_group(g)[metric_col].mean() for g in ORDER]
    stds = [grp.get_group(g)[metric_col].std() for g in ORDER]
    xs = np.arange(len(ORDER))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors, zorder=3)
    # jittered per-sample points
    for i, g in enumerate(ORDER):
        vals = grp.get_group(g)[metric_col].values
        jit = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, color="k", s=18,
                   zorder=4, alpha=0.7)
    # annotate mean (and absolute mean if provided) centered inside each bar
    for i, g in enumerate(ORDER):
        txt = f"{means[i]:+.1f}%"
        if absolute_col is not None:
            txt += f"\n({grp.get_group(g)[absolute_col].mean()*1000:+.1f} mg)"
        ax.text(i, means[i] * 0.5, txt, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.35, ec="none"))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([g.replace(" (", "\n(") for g in ORDER], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    print(f"saved {fname}")


grouped_bar(
    "wet_change_pct",
    "Wet-weight change, decell → Chem Pol (%)",
    "Wet-weight change across polymerization",
    "wet_weight_change_by_group.png",
    absolute_col="wet_change",
)
grouped_bar(
    "dry_change_pct",
    "Dry-mass change, decell → Chem Pol (%)",
    "Dry-weight (PPy uptake) change across polymerization",
    "dry_weight_change_by_group.png",
    absolute_col="dry_change",
)
