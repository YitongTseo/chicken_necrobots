"""
Pretension comparison - 2026-07-20 1000 um decell chicken + PPy, LENGTH channel, 2.5 mV/s.

Three LENGTH runs on the SAME sample at three different pretensions:

  4.13 V  (one cycle)   logged 18:11
  4.9  V  (three cycles, the original run)  logged 16:21
  5.25 V  (one cycle)   logged 17:43

Pretension force uses the same gain as the CV force channel, referenced to the
no-load baseline of 1.25 V:   F_pre = (V_pre - 1.25) * 56.7 mN

Displacement = CH1 * -0.4130 mm/V (1x gain).

Runs happened in the order 4.9 -> 5.25 -> 4.13, so elapsed time since electropolymerization
is CONFOUNDED with pretension here; see the printed summary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SCAN_RATE_V_PER_S = 0.0025
V_HI, V_LO = 1.0, -1.0
LEG_S = (V_HI - V_LO) / SCAN_RATE_V_PER_S      # 800 s
PERIOD_S = 2 * LEG_S                            # 1600 s

FORCE_GAIN_MN_PER_V = 56.7
NO_LOAD_BASELINE_V = 1.25
LENGTH_GAIN_MM_PER_V = -0.4130

WINDOW, K = 41, 5.0
SMOOTH_WINDOW, SAVGOL_POLY = 101, 3

RUNS = [
    dict(tag="4.13 V", v_pre=4.13, order=3, cycle_used=0, color="#2b6cb0",
         path="07.20.26_1000um_chicken_LENGTH_4.1Vpretension_1_cycle_2.5mVpersec_scope_log_20260720_181119.csv"),
    dict(tag="4.90 V", v_pre=4.90, order=1, cycle_used=0, color="#38a169",
         path="07.20.26_1000um_chicken_LENGTH_2.5mVpersec_scope_log_20260720_162118.csv"),
    dict(tag="5.25 V", v_pre=5.25, order=2, cycle_used=0, color="#e53e3e",
         path="07.20.26_1000um_chicken_LENGTH_5.25Vpretension_1_cycle_2.5mVpersec_scope_log_20260720_174328.csv"),
]


def applied_potential(t):
    ph = np.asarray(t, float) % PERIOD_S
    return np.where(ph <= LEG_S,
                    V_HI - SCAN_RATE_V_PER_S * ph,
                    V_LO + SCAN_RATE_V_PER_S * (ph - LEG_S))


def mark_outliers(x, window=WINDOW, k=K):
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    ad = (s - med).abs()
    mad = ad.rolling(window, center=True, min_periods=1).median()
    mad = mad.clip(lower=max(np.nanstd(x) * 1e-3, 1e-9))
    return (ad > k * mad).values


def smooth(y, window=SMOOTH_WINDOW, poly=SAVGOL_POLY):
    w = min(window, len(y))
    w -= (w % 2 == 0)
    return savgol_filter(y, w, poly) if w > poly else np.asarray(y, float)


def load_cycle(run, cycle):
    """Return one cycle of a run as a tidy frame, in micrometres."""
    d = pd.read_csv(run["path"]).rename(columns={"elapsed_s": "t"})
    lo, hi = cycle * PERIOD_S, (cycle + 1) * PERIOD_S
    d = d[(d["t"] >= lo) & (d["t"] < hi)].copy()
    d = d.loc[~mark_outliers(d["ch1_MEAN_V"].values)].copy()

    d["v_applied"] = applied_potential(d["t"].values)
    d["t_cyc"] = d["t"] - lo
    d["disp_um"] = d["ch1_MEAN_V"] * LENGTH_GAIN_MM_PER_V * 1000.0
    d["sm"] = smooth(d["disp_um"].values)

    # remove linear-in-time creep so the loop closes on itself
    t = d["t_cyc"].values
    sm = d["sm"].values
    slope = (sm[-1] - sm[0]) / (t[-1] - t[0])
    d["closed"] = d["disp_um"].values - slope * (t - t[0])
    d["sm_closed"] = smooth(d["closed"].values)
    d.attrs["creep_um"] = slope * (t[-1] - t[0])
    return d


def metrics(d):
    """Per-cycle actuation + roughness metrics.

    'reversals' (direction changes of the smoothed trace) is deliberately NOT used:
    the 4.90 V run gives 6 / 27 / 32 across its own three cycles at identical
    pretension, so it measures noise rather than bumpiness. Total variation
    normalised by swing is the stable roughness measure.
    """
    y = d["sm_closed"].values - d["sm_closed"].values[0]
    resid = d["closed"].values - d["sm_closed"].values
    swing = y.max() - y.min()
    tv = np.abs(np.diff(d["sm_closed"].values)).sum()
    return dict(
        rest_disp_um=d["disp_um"].mean(),
        swing_um=swing,
        creep_um=d.attrs["creep_um"],
        noise_rms_um=resid.std(),
        noise_pct_of_swing=100 * resid.std() / swing,
        tv_ratio=tv / swing,
        n_pts=len(d),
    )


rows, data, per_cycle = [], {}, []
for run in RUNS:
    cycles = [0, 1, 2] if run["tag"] == "4.90 V" else [0]
    for c in cycles:
        d = load_cycle(run, c)
        m = metrics(d)
        per_cycle.append(dict(tag=run["tag"], cycle=c + 1, **m))
        if c == run["cycle_used"]:
            data[run["tag"]] = d
    # headline row uses the LAST available cycle: for the 3-cycle run that is a
    # conditioned cycle, comparable to the single-cycle runs which were themselves
    # recorded on already-conditioned tissue.
    d_rep = load_cycle(run, cycles[-1])
    rows.append(dict(
        pretension_V=run["v_pre"],
        pretension_mN=(run["v_pre"] - NO_LOAD_BASELINE_V) * FORCE_GAIN_MN_PER_V,
        run_order=run["order"],
        cycle_reported=cycles[-1] + 1,
        **metrics(d_rep),
    ))

summary = pd.DataFrame(rows).sort_values("pretension_V").reset_index(drop=True)
pc = pd.DataFrame(per_cycle)
pd.set_option("display.width", 220, "display.max_columns", 25)

print("=== Per-cycle metrics (all cycles of all runs) ===")
print(pc.round(3).to_string(index=False))

print("\n=== Headline comparison (last cycle of each run = conditioned state) ===")
print(summary.round(3).to_string(index=False))

# The 4.90 V run is the only repeat at a FIXED pretension, so its cycle-to-cycle
# spread is the natural yardstick for whether between-pretension gaps mean anything.
ref = pc[pc["tag"] == "4.90 V"]
print("\n=== Within-run spread at fixed pretension (4.90 V, n=3 cycles) ===")
for col in ["swing_um", "creep_um", "noise_rms_um", "tv_ratio"]:
    lo, hi = ref[col].min(), ref[col].max()
    between = summary[col].max() - summary[col].min()
    print(f"  {col:16s} within-run {lo:8.3f} .. {hi:8.3f} (range {hi-lo:7.3f})   "
          f"between-pretension range {between:7.3f}   "
          f"{'SEPARABLE' if between > (hi - lo) else 'NOT separable'}")

# ============================== FIGURES =====================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

# (a) raw traces vs time, each offset to its own start
ax = axes[0, 0]
for run in RUNS:
    d = data[run["tag"]]
    ax.plot(d["t_cyc"], d["disp_um"] - d["disp_um"].values[0], lw=0.6, alpha=0.35,
            color=run["color"])
    ax.plot(d["t_cyc"], d["sm"] - d["sm"].values[0], lw=2.0, color=run["color"],
            label=f"{run['tag']}  ({(run['v_pre']-NO_LOAD_BASELINE_V)*FORCE_GAIN_MN_PER_V:.0f} mN)")
ax.axvline(LEG_S, color="k", ls="--", alpha=0.5, label="-1 V turning point")
ax.set_xlabel("time within cycle (s)")
ax.set_ylabel("displacement from cycle start (um)")
ax.set_title("(a) Raw displacement vs time (thin = unsmoothed)", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# (b) CV loops
ax = axes[0, 1]
for run in RUNS:
    d = data[run["tag"]]
    y = d["sm_closed"].values - d["sm_closed"].values[0]
    ax.plot(d["v_applied"], y, lw=2.0, color=run["color"], label=run["tag"])
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.invert_xaxis()
ax.set_xlabel("Applied potential (V)")
ax.set_ylabel("delta displacement from +1 V start (um)")
ax.set_title("(b) Displacement CV loops (creep-corrected)", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# (c) swing vs pretension, against the within-run spread at fixed pretension
ax = axes[1, 0]
lo, hi = ref["swing_um"].min(), ref["swing_um"].max()
ax.axhspan(lo, hi, color="0.75", alpha=0.45, zorder=1,
           label=f"4.90 V within-run range\n({lo:.1f}-{hi:.1f} um, n=3 cycles)")
for _, r in pc.iterrows():
    v = [x for x in RUNS if x["tag"] == r["tag"]][0]
    ax.scatter((v["v_pre"] - NO_LOAD_BASELINE_V) * FORCE_GAIN_MN_PER_V, r["swing_um"],
               s=52, color=v["color"], alpha=0.55, edgecolor="white", zorder=3)
ax.plot(summary["pretension_mN"], summary["swing_um"], "-o", color="#1a202c",
        lw=2, ms=10, markeredgecolor="k", zorder=4, label="last (conditioned) cycle")
for _, r in summary.iterrows():
    ax.annotate(f"{r['pretension_V']:.2f} V\n(run #{int(r['run_order'])})",
                (r["pretension_mN"], r["swing_um"]), textcoords="offset points",
                xytext=(9, -18), fontsize=8)
ax.set_xlabel("pretension (mN)")
ax.set_ylabel("actuation swing (um)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="best")
ax.set_title("(c) Swing vs pretension - between-pretension gap sits\n"
             "inside the single-pretension cycle-to-cycle spread", fontsize=9.5)

# (d) noise residuals
ax = axes[1, 1]
for run in RUNS:
    d = data[run["tag"]]
    resid = d["closed"].values - d["sm_closed"].values
    ax.plot(d["t_cyc"], resid, lw=0.7, alpha=0.75, color=run["color"],
            label=f"{run['tag']}  RMS {resid.std():.3f} um")
ax.set_xlabel("time within cycle (s)")
ax.set_ylabel("residual after smoothing (um)")
ax.set_title("(d) High-frequency noise ('bumpiness')", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("LENGTH pretension comparison - 2026-07-20 1000 um decell chicken + PPy, 2.5 mV/s",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("cv_20260720_pretension_comparison.png", bbox_inches="tight")
print("\nsaved cv_20260720_pretension_comparison.png")
