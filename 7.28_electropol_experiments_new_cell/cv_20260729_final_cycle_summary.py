"""
Single summary figure: FINAL CYCLE (cycle 3) only, for the 2026-07-29 1000 um ALONG sample.

Four panels - the mechanical CV from the FORCE run, the mechanical CV from the LENGTH run,
and the potentiostat i-vs-E CV recorded simultaneously with each. Cycle 3 is the one to
show: cycles 2 and 3 overlay, so the first-scan PPy break-in is behind us.

Same pipeline as cv_20260729_along_with_current.py (MAD outlier filter, per-cycle
Savitzky-Golay, per-cycle linear creep removal), measured potential from the potentiostat
export, measured geometry from 2026-07-30.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SCAN_RATE_V_PER_S = 0.0025
V_HI, V_LO = 1.0, -1.0
LEG_S = (V_HI - V_LO) / SCAN_RATE_V_PER_S
PERIOD_S = 2 * LEG_S
N_CYCLES = 3
TOTAL_S = N_CYCLES * PERIOD_S
LAST = N_CYCLES - 1

FORCE_GAIN_MN_PER_V = 56.7
LENGTH_GAIN_MM_PER_V = -0.4130
L0_MM, WIDTH_MM, THICKNESS_MM = 4.0, 5.0, 0.2230
AREA_MM2 = WIDTH_MM * THICKNESS_MM

WINDOW, K = 41, 5.0
SMOOTH_WINDOW, SAVGOL_POLY = 101, 3

C_FORCE, C_LENGTH, C_CURRENT = "#c53030", "#2b6cb0", "#b7791f"

RUNS = {
    "FORCE": dict(
        scope="1000um_ALONG_FORCE_scope_log_20260729_150114.csv",
        pot="potentiostat/07-29-26_FORCE_2.5mvpersec_1000um_ALONG.csv",
        channel="ch2_MEAN_V", gain=FORCE_GAIN_MN_PER_V, unit="mN", label="Force",
        colour=C_FORCE),
    "LENGTH": dict(
        scope="1000um_ALONG_LENGTH_scope_log_20260729_162518.csv",
        pot="potentiostat/07-29-26_LENGTH_2.5mvpersec_1000um_ALONG.csv",
        channel="ch1_MEAN_V", gain=LENGTH_GAIN_MM_PER_V, unit="mm",
        label="Displacement", colour=C_LENGTH),
}


def load_potentiostat(path):
    raw = pd.read_csv(path, encoding="utf-16", skiprows=5).dropna(axis=1, how="all")
    frames = []
    for s in range(N_CYCLES):
        V = pd.to_numeric(raw.iloc[:, 2 * s], errors="coerce").values
        I = pd.to_numeric(raw.iloc[:, 2 * s + 1], errors="coerce").values
        m = ~np.isnan(V) & ~np.isnan(I)
        V, I = V[m], I[m]
        dt = np.abs(np.diff(V, prepend=V[0])) / SCAN_RATE_V_PER_S
        frames.append(pd.DataFrame(dict(t=np.cumsum(dt) + s * PERIOD_S, v=V, i_uA=I,
                                        scan=s)))
    return pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)


def mark_outliers(x, window=WINDOW, k=K):
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    ad = (s - med).abs()
    mad = ad.rolling(window, center=True, min_periods=1).median()
    mad = mad.clip(lower=max(np.nanstd(x) * 1e-3, 1e-9))
    return (ad > k * mad).values


def savgol_per_cycle(df, col):
    out = pd.Series(np.nan, index=df.index)
    for _, sub in df.groupby("cycle"):
        sub = sub.sort_values("t")
        y = sub[col].values
        w = min(SMOOTH_WINDOW, len(y))
        w -= (w % 2 == 0)
        out.loc[sub.index] = savgol_filter(y, w, SAVGOL_POLY) if w > SAVGOL_POLY else y
    return out


def close_loop_per_cycle(df, col, scol):
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
    p = load_potentiostat(run["pot"])
    d = pd.read_csv(run["scope"]).rename(columns={"elapsed_s": "t"})
    d = d[(d["t"] >= 0) & (d["t"] <= TOTAL_S)].copy()
    d = d.loc[~mark_outliers(d[run["channel"]].values)].copy()
    d["cycle"] = np.clip((d["t"].values // PERIOD_S).astype(int), 0, N_CYCLES - 1)
    d["signal"] = d[run["channel"]] * run["gain"]
    d["smooth"] = savgol_per_cycle(d, "signal")
    d["closed"] = close_loop_per_cycle(d, "signal", "smooth")
    d["smooth_closed"] = savgol_per_cycle(d, "closed")
    d["v_meas"] = np.interp(d["t"], p["t"], p["v"])
    return d, p


def arrows(ax, x, y, frac=(0.18, 0.42, 0.72), colour="k"):
    """Small direction arrows so the sweep sense is readable."""
    n = len(x)
    for f in frac:
        i = int(f * n)
        if 0 < i < n - 2:
            ax.annotate("", xy=(x[i + 2], y[i + 2]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4,
                                        mutation_scale=15, alpha=0.85))


plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))

stats = {}
for col, (tag, run) in enumerate(RUNS.items()):
    d, p = load(run)
    s = d[d["cycle"] == LAST].sort_values("t")
    v = s["v_meas"].values
    y = s["smooth_closed"].values - s["smooth_closed"].values[0]
    swing = y.max() - y.min()
    norm = swing / AREA_MM2 / 1000.0 if tag == "FORCE" else swing / L0_MM * 100.0
    nunit = "MPa" if tag == "FORCE" else "%"
    stats[tag] = (swing, norm, nunit, v[int(np.argmax(y))], v[int(np.argmin(y))])

    # --- top row: mechanical CV -------------------------------------------
    ax = axes[0, col]
    ax.plot(v, y, lw=2.2, color=run["colour"])
    arrows(ax, v, y, colour=run["colour"])
    ax.scatter([v[0]], [y[0]], s=70, color=run["colour"], ec="k", lw=0.6, zorder=6,
               label="start / end (+1 V)")
    ax.scatter([v[int(np.argmax(y))], v[int(np.argmin(y))]], [y.max(), y.min()],
               marker="x", s=80, color="k", lw=1.6, zorder=6, label="extrema")
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.invert_xaxis()
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel(f"$\\Delta$ {run['label'].lower()} ({run['unit']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{run['label']} CV - final cycle\nswing {swing:.3g} {run['unit']} "
                 f"= {norm:.3g} {nunit}", fontsize=10.5, fontweight="bold",
                 color=run["colour"])
    # secondary axis in normalised units
    conv = (1 / AREA_MM2 / 1000.0) if tag == "FORCE" else (1 / L0_MM * 100.0)
    sec = ax.secondary_yaxis("right", functions=(lambda a, c=conv: a * c,
                                                 lambda a, c=conv: a / c))
    sec.set_ylabel("stress (MPa)" if tag == "FORCE" else "strain (%)", fontsize=9)

    # --- bottom row: simultaneous potentiostat CV --------------------------
    sp = p[p["scan"] == LAST]
    vi, ii = sp["v"].values, sp["i_uA"].values / 1000.0
    axi = axes[1, col]
    axi.plot(vi, ii, lw=1.6, color=C_CURRENT)
    arrows(axi, vi, ii, colour=C_CURRENT)
    axi.axhline(0, color="k", lw=0.5, alpha=0.6)
    axi.invert_xaxis()
    axi.set_xlabel("Potential (V)")
    axi.set_ylabel("Current (mA)")
    axi.grid(alpha=0.3)
    axi.set_title(f"Potentiostat CV during the {run['label'].lower()} run - final cycle\n"
                  f"peak |i| {np.abs(ii).max():.2f} mA", fontsize=10.5)

fig.suptitle("2026-07-29 1000 um ALONG + PPy, new electropol cell - final cycle (3 of 3), "
             "2.5 mV/s, +1 V $\\rightarrow$ -1 V $\\rightarrow$ +1 V\n"
             f"L0 {L0_MM} mm, {WIDTH_MM} $\\times$ {THICKNESS_MM} mm "
             f"({AREA_MM2:.3f} mm$^2$); mechanical traces creep-corrected",
             fontweight="bold", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("cv_20260729_final_cycle_summary.png", bbox_inches="tight")
print("saved cv_20260729_final_cycle_summary.png\n")
for tag, (sw, nm, nu, vmax, vmin) in stats.items():
    print(f"{tag:6s} final cycle: swing {sw:.4f} -> {nm:.4f} {nu}; "
          f"max at {vmax:+.3f} V, min at {vmin:+.3f} V")
