"""
CV analysis - 2026-07-29 1000 um "ALONG" sample, new electropol cell.

FORCE and LENGTH runs, same pipeline as
7.15.26_thin_ppy_chicken_experiments/cv_20260720_force_length_analysis.py:
no potentiostat export, so the applied potential is synthesized from the protocol, and
scope + potentiostat start together (t0 = 0 for each file - no phase fitting).

Protocol confirmed from the data rather than assumed: autocorrelation of both traces
peaks at ~1570-1610 s, matching 2.5 mV/s over +1 V -> -1 V -> +1 V (800 s per leg,
1600 s per cycle, 3 cycles = 4800 s). Both logs overrun the sweep by a few seconds;
everything past 4800 s is discarded.

  FORCE  run -> CH2, 56.7 mN/V     (CH1 is 0)
  LENGTH run -> CH1, -0.4130 mm/V  (CH2 is 0)

GEOMETRY IS BORROWED, NOT MEASURED: stress and strain here use the 2026-07-20 1000 um
sample's dimensions (L0 3.2 mm, 4.0 x 0.27 mm, area 1.08 mm^2) as a stand-in, at the
user's request, so the normalised numbers are order-of-magnitude only. The raw mN and
mm swings are the trustworthy quantities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SCAN_RATE_V_PER_S = 0.0025
V_HI, V_LO = 1.0, -1.0
LEG_S = (V_HI - V_LO) / SCAN_RATE_V_PER_S        # 800 s per leg
PERIOD_S = 2 * LEG_S                              # 1600 s per cycle
N_CYCLES = 3
TOTAL_S = N_CYCLES * PERIOD_S                     # 4800 s

FORCE_GAIN_MN_PER_V = 56.7        # 1x gain
LENGTH_GAIN_MM_PER_V = -0.4130    # 1x gain

# Borrowed from the 2026-07-20 1000 um sample - NOT measured for this sample.
L0_MM, WIDTH_MM, THICKNESS_MM = 3.2, 4.0, 0.27
AREA_MM2 = WIDTH_MM * THICKNESS_MM

WINDOW, K = 41, 5.0
SMOOTH_WINDOW, SAVGOL_POLY = 101, 3
PALETTE = ["#1f77b4", "#d62728", "#2ca02c"]

RUNS = {
    "FORCE": dict(
        path="1000um_ALONG_FORCE_scope_log_20260729_150114.csv",
        channel="ch2_MEAN_V", gain=FORCE_GAIN_MN_PER_V,
        unit="mN", label="Force", norm_unit="MPa",
    ),
    "LENGTH": dict(
        path="1000um_ALONG_LENGTH_scope_log_20260729_162518.csv",
        channel="ch1_MEAN_V", gain=LENGTH_GAIN_MM_PER_V,
        unit="mm", label="Displacement", norm_unit="%",
    ),
}


def normalise(tag, swing):
    """Force swing -> stress (MPa); displacement swing -> strain (%)."""
    return swing / AREA_MM2 / 1000.0 if tag == "FORCE" else swing / L0_MM * 100.0


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


def savgol_per_cycle(df, col, window=SMOOTH_WINDOW, poly=SAVGOL_POLY):
    out = pd.Series(np.nan, index=df.index)
    for _, sub in df.groupby("cycle"):
        sub = sub.sort_values("t")
        y = sub[col].values
        if len(y) < 3:
            out.loc[sub.index] = y
            continue
        w = min(window, len(y))
        w -= (w % 2 == 0)
        out.loc[sub.index] = savgol_filter(y, w, poly) if w > poly else y
    return out


def close_loop_per_cycle(df, col, scol):
    """Remove per-cycle linear-in-time creep so each cycle ends where it began."""
    out = df[col].astype(float).copy()
    for _, sub in df.groupby("cycle"):
        sub = sub.sort_values("t")
        t, y = sub["t"].values, sub[scol].values
        if len(sub) < 2 or t[-1] == t[0]:
            continue
        slope = (y[-1] - y[0]) / (t[-1] - t[0])
        out.loc[sub.index] = df.loc[sub.index, col].values - slope * (t - t[0])
    return out


def load(run):
    d = pd.read_csv(run["path"]).rename(columns={"elapsed_s": "t"})
    n_raw = len(d)
    d = d[(d["t"] >= 0) & (d["t"] <= TOTAL_S)].copy()
    dropped_tail = n_raw - len(d)

    mask = mark_outliers(d[run["channel"]].values)
    d = d.loc[~mask].copy()

    d["v_applied"] = applied_potential(d["t"].values)
    d["cycle"] = np.clip((d["t"].values // PERIOD_S).astype(int), 0, N_CYCLES - 1)
    d["leg"] = np.where(d["t"].values % PERIOD_S <= LEG_S, "cathodic", "anodic")

    d["signal"] = d[run["channel"]] * run["gain"]
    d["smooth"] = savgol_per_cycle(d, "signal")
    d["closed"] = close_loop_per_cycle(d, "signal", "smooth")
    d["smooth_closed"] = savgol_per_cycle(d, "closed")

    print(f"{run['label']}: {n_raw} raw pts, {dropped_tail} past {TOTAL_S:.0f}s dropped, "
          f"{int(mask.sum())} outliers removed, {len(d)} kept")
    return d


def plot_run(tag, run, d):
    unit, lab = run["unit"], run["label"]
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, N_CYCLES, height_ratios=[1, 1.15], hspace=0.32, wspace=0.26)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(d["t"], d["signal"], lw=0.5, color="0.75", label=f"raw {lab.lower()}")
    for c in range(N_CYCLES):
        s = d[d["cycle"] == c].sort_values("t")
        ax.plot(s["t"], s["smooth"], lw=1.8, color=PALETTE[c], label=f"cycle {c+1}")
    for b in np.arange(0, TOTAL_S + 1, LEG_S):
        ax.axvline(b, color="k", ls=":", alpha=0.35, lw=0.9)
    ax.set(xlabel="elapsed time (s)", ylabel=f"{lab} ({unit})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5)
    ax.set_title(f"{lab} vs time  (dotted = sweep turning points every {LEG_S:.0f} s)",
                 fontsize=10)
    axv = ax.twinx()
    tg = np.linspace(0, TOTAL_S, 2000)
    axv.plot(tg, applied_potential(tg), color="k", lw=1.0, alpha=0.35)
    axv.set_ylabel("applied potential (V)", color="0.35")
    axv.tick_params(axis="y", colors="0.35")

    for c in range(N_CYCLES):
        axc = fig.add_subplot(gs[1, c])
        s = d[d["cycle"] == c].sort_values("t")
        if len(s) < 2:
            continue
        y_open = s["smooth"].values - s["smooth"].values[0]
        y_closed = s["smooth_closed"].values - s["smooth_closed"].values[0]
        swing = y_closed.max() - y_closed.min()

        axc.plot(s["v_applied"], y_open, color=PALETTE[c], lw=1.1, ls=":", alpha=0.55,
                 label="original (open)")
        axc.plot(s["v_applied"], y_closed, color=PALETTE[c], lw=2.0,
                 label="creep-corrected")
        axc.scatter([s["v_applied"].values[0], s["v_applied"].values[-1]],
                    [y_closed[0], y_closed[-1]], color=PALETTE[c], s=45, ec="k",
                    lw=0.5, zorder=5)
        axc.axhline(0, color="k", lw=0.5, alpha=0.5)
        axc.invert_xaxis()
        axc.set_xlabel("Applied potential (V)")
        if c == 0:
            axc.set_ylabel(f"delta {lab.lower()} from +1 V start ({unit})")
        axc.grid(alpha=0.3)
        axc.legend(fontsize=7.5)
        axc.set_title(f"cycle {c+1}   (swing {swing:.3g} {unit} = "
                      f"{normalise(tag, swing):.3g} {run['norm_unit']})", fontsize=10)

    fig.suptitle(f"2026-07-29 1000 um ALONG + PPy, new electropol cell - {lab} CV, "
                 "2.5 mV/s (+1 V -> -1 V -> +1 V, 3 cycles)", fontweight="bold")
    out = f"cv_20260729_along_{tag.lower()}_cv.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"  saved {out}")


def overlay(tag, run, d):
    unit, lab = run["unit"], run["label"]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for c in range(N_CYCLES):
        s = d[d["cycle"] == c].sort_values("t")
        if len(s) < 2:
            continue
        y = s["smooth_closed"].values - s["smooth_closed"].values[0]
        ax.plot(s["v_applied"], y, color=PALETTE[c], lw=1.9, label=f"cycle {c+1}")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.invert_xaxis()
    ax.set(xlabel="Applied potential (V)",
           ylabel=f"delta {lab.lower()} from +1 V start ({unit})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{lab} CV - all cycles overlaid\n(creep-corrected, 2.5 mV/s)",
                 fontsize=11, fontweight="bold")
    out = f"cv_20260729_along_{tag.lower()}_overlay.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"  saved {out}")


plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
summary = []
for tag, run in RUNS.items():
    d = load(run)
    plot_run(tag, run, d)
    overlay(tag, run, d)
    for c in range(N_CYCLES):
        s = d[d["cycle"] == c].sort_values("t")
        y = s["smooth_closed"].values - s["smooth_closed"].values[0]
        raw = s["smooth"].values
        swing = y.max() - y.min()
        summary.append(dict(run=tag, cycle=c + 1, unit=run["unit"], swing=swing,
                            normalised=normalise(tag, swing),
                            norm_unit=run["norm_unit"],
                            net_creep=raw[-1] - raw[0],
                            v_at_max=s["v_applied"].values[int(np.argmax(y))],
                            v_at_min=s["v_applied"].values[int(np.argmin(y))]))

print(f"\n=== BORROWED geometry (2026-07-20 sample): L0={L0_MM} mm, W={WIDTH_MM} mm, "
      f"T={THICKNESS_MM} mm, area={AREA_MM2:.3f} mm^2 ===")
print("\n=== per-cycle actuation swing (creep-corrected) ===")
print(pd.DataFrame(summary).round(4).to_string(index=False))
