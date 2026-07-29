"""
CV analysis - 2026-07-29 1000 um "ALONG" sample, new electropol cell.
REVISION: now uses the logged potentiostat potential + current instead of a synthesized
triangle, and the measured sample geometry instead of the 07.20 stand-in.

Potentiostat exports (UTF-16, 5 header rows, three "CV i vs E Scan n" column pairs V/uA)
confirm the protocol exactly: 4034 points per scan at 1.015 mV per point = 1600.0 s per
scan, turning point at precisely 50% (index 2017), 3 scans = 4800 s. So the previously
synthesized +1 -> -1 -> +1 V triangle at 2.5 mV/s was correct; what is new is the CURRENT,
which lets us (a) verify the scope/potentiostat time alignment empirically for the first
time and (b) plot force against charge.

Timing: the potentiostat header's "Date and time measurement" sits 493.2 s after the scope
log's first timestamp in BOTH runs (to <0.1 s), which is clock skew between the two
machines, not a real delay - a human starting two instruments could not reproduce an
offset to a tenth of a second 1.5 h apart. The scope logs are also only 4806 / 4827 s
long, so a genuine 493 s delay would mean the last ~490 s of each 4800 s CV was never
recorded. t0 = 0 stands, and the charge cross-correlation below tests it directly.

Geometry MEASURED 2026-07-30: W = 5.0 mm, L0 = 4.0 mm ("4 mm tall"), T = 0.2230 mm.
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

FORCE_GAIN_MN_PER_V = 56.7
LENGTH_GAIN_MM_PER_V = -0.4130
NO_LOAD_V = 1.25

# Measured 2026-07-30. "4 mm tall" is taken as the free gauge length L0.
L0_MM, WIDTH_MM, THICKNESS_MM = 4.0, 5.0, 0.2230
AREA_MM2 = WIDTH_MM * THICKNESS_MM

WINDOW, K = 41, 5.0
SMOOTH_WINDOW, SAVGOL_POLY = 101, 3
PALETTE = ["#1f77b4", "#d62728", "#2ca02c"]

RUNS = {
    "FORCE": dict(
        scope="1000um_ALONG_FORCE_scope_log_20260729_150114.csv",
        pot="potentiostat/07-29-26_FORCE_2.5mvpersec_1000um_ALONG.csv",
        channel="ch2_MEAN_V", gain=FORCE_GAIN_MN_PER_V,
        unit="mN", label="Force", norm_unit="MPa",
    ),
    "LENGTH": dict(
        scope="1000um_ALONG_LENGTH_scope_log_20260729_162518.csv",
        pot="potentiostat/07-29-26_LENGTH_2.5mvpersec_1000um_ALONG.csv",
        channel="ch1_MEAN_V", gain=LENGTH_GAIN_MM_PER_V,
        unit="mm", label="Displacement", norm_unit="%",
    ),
}


def normalise(tag, swing):
    return swing / AREA_MM2 / 1000.0 if tag == "FORCE" else swing / L0_MM * 100.0


def load_potentiostat(path):
    """Return a frame with t, v, i_uA, q_mC built from the three CV scans."""
    raw = pd.read_csv(path, encoding="utf-16", skiprows=5).dropna(axis=1, how="all")
    frames = []
    for s in range(N_CYCLES):
        V = pd.to_numeric(raw.iloc[:, 2 * s], errors="coerce").values
        I = pd.to_numeric(raw.iloc[:, 2 * s + 1], errors="coerce").values
        m = ~np.isnan(V) & ~np.isnan(I)
        V, I = V[m], I[m]
        # time from the potential itself: each step is |dV| / scan rate
        dt = np.abs(np.diff(V, prepend=V[0])) / SCAN_RATE_V_PER_S
        t = np.cumsum(dt) + s * PERIOD_S
        frames.append(pd.DataFrame(dict(t=t, v=V, i_uA=I, scan=s)))
    p = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    # charge in mC, integrated over the whole run
    p["q_mC"] = np.concatenate([[0.0], np.cumsum(
        0.5 * (p["i_uA"].values[1:] + p["i_uA"].values[:-1]) * np.diff(p["t"].values))]) / 1e3
    return p


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
    out = df[col].astype(float).copy()
    for _, sub in df.groupby("cycle"):
        sub = sub.sort_values("t")
        t, y = sub["t"].values, sub[scol].values
        if len(sub) < 2 or t[-1] == t[0]:
            continue
        slope = (y[-1] - y[0]) / (t[-1] - t[0])
        out.loc[sub.index] = df.loc[sub.index, col].values - slope * (t - t[0])
    return out


def alignment_lag(t, sig, p, lags=np.arange(-700, 701, 5.0)):
    """Force should track charge (strain ~ charge in PPy). Cross-correlate the two,
    each linearly detrended, and report the lag that maximises |r|."""
    tg = np.arange(200.0, TOTAL_S - 200.0, 2.0)
    s = np.interp(tg, t, sig)
    s = s - np.polyval(np.polyfit(tg, s, 1), tg)
    best, curve = (np.nan, 0.0), []
    for L in lags:
        q = np.interp(tg + L, p["t"].values, p["q_mC"].values)
        q = q - np.polyval(np.polyfit(tg, q, 1), tg)
        r = float(np.corrcoef(s, q)[0, 1])
        curve.append(r)
        if np.isfinite(r) and abs(r) > abs(best[1]):
            best = (L, r)
    return best, np.array(curve), lags


def load(tag, run):
    p = load_potentiostat(run["pot"])
    d = pd.read_csv(run["scope"]).rename(columns={"elapsed_s": "t"})
    n_raw = len(d)
    d = d[(d["t"] >= 0) & (d["t"] <= TOTAL_S)].copy()
    dropped = n_raw - len(d)

    mask = mark_outliers(d[run["channel"]].values)
    d = d.loc[~mask].copy()

    d["cycle"] = np.clip((d["t"].values // PERIOD_S).astype(int), 0, N_CYCLES - 1)
    d["signal"] = d[run["channel"]] * run["gain"]
    d["smooth"] = savgol_per_cycle(d, "signal")
    d["closed"] = close_loop_per_cycle(d, "signal", "smooth")
    d["smooth_closed"] = savgol_per_cycle(d, "closed")

    # MEASURED potential and current, interpolated onto the scope's timebase
    d["v_meas"] = np.interp(d["t"], p["t"], p["v"])
    d["i_uA"] = np.interp(d["t"], p["t"], p["i_uA"])
    d["q_mC"] = np.interp(d["t"], p["t"], p["q_mC"])
    d["leg"] = np.where(d["t"].values % PERIOD_S <= LEG_S, "cathodic", "anodic")

    (lag, r), curve, lags = alignment_lag(d["t"].values, d["smooth"].values, p)
    r0 = curve[int(np.argmin(np.abs(lags)))]
    print(f"{run['label']}: {n_raw} raw pts, {dropped} past {TOTAL_S:.0f}s dropped, "
          f"{int(mask.sum())} outliers removed, {len(d)} kept")
    print(f"  alignment check ({run['label']} vs charge): best lag {lag:+.0f} s "
          f"(r={r:+.3f}); at lag 0 r={r0:+.3f}")
    print(f"  total charge passed {p['q_mC'].iloc[-1]:.1f} mC; "
          f"|I| max {p['i_uA'].abs().max():.0f} uA")
    return d, p, (lag, r, r0, curve, lags)


def plot_run(tag, run, d, p, align):
    unit, lab = run["unit"], run["label"]
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.85, 1.15, 1.0], hspace=0.42,
                          wspace=0.27)

    # --- row 1: signal vs time with measured potential -----------------------
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
    ax.set_title(f"{lab} vs time, with MEASURED potential (grey)", fontsize=10)
    axv = ax.twinx()
    axv.plot(p["t"], p["v"], color="k", lw=1.0, alpha=0.35)
    axv.set_ylabel("measured potential (V)", color="0.35")
    axv.tick_params(axis="y", colors="0.35")

    # --- row 2: current and charge vs time -----------------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.plot(p["t"], p["i_uA"] / 1000.0, lw=0.8, color="#b7791f")
    ax.axhline(0, color="k", lw=0.5)
    for b in np.arange(0, TOTAL_S + 1, LEG_S):
        ax.axvline(b, color="k", ls=":", alpha=0.35, lw=0.9)
    ax.set(xlabel="elapsed time (s)", ylabel="current (mA)")
    ax.grid(alpha=0.3)
    ax.set_title("Potentiostat current (gold) and cumulative charge (purple)", fontsize=10)
    axq = ax.twinx()
    axq.plot(p["t"], p["q_mC"], color="#6b46c1", lw=1.5)
    axq.set_ylabel("charge (mC)", color="#6b46c1")
    axq.tick_params(axis="y", colors="#6b46c1")

    # --- row 3: mechanical CV loop per cycle, vs measured potential ----------
    for c in range(N_CYCLES):
        axc = fig.add_subplot(gs[2, c])
        s = d[d["cycle"] == c].sort_values("t")
        y_open = s["smooth"].values - s["smooth"].values[0]
        y_closed = s["smooth_closed"].values - s["smooth_closed"].values[0]
        swing = y_closed.max() - y_closed.min()
        axc.plot(s["v_meas"], y_open, color=PALETTE[c], lw=1.1, ls=":", alpha=0.55,
                 label="original (open)")
        axc.plot(s["v_meas"], y_closed, color=PALETTE[c], lw=2.0, label="creep-corrected")
        axc.axhline(0, color="k", lw=0.5, alpha=0.5)
        axc.invert_xaxis()
        axc.set_xlabel("measured potential (V)")
        if c == 0:
            axc.set_ylabel(f"delta {lab.lower()} ({unit})")
        axc.grid(alpha=0.3)
        axc.legend(fontsize=7.5)
        axc.set_title(f"cycle {c+1}  (swing {swing:.3g} {unit} = "
                      f"{normalise(tag, swing):.3g} {run['norm_unit']})", fontsize=9.5)

    # --- row 4: electrochemical CV, signal vs charge, alignment curve -------
    axe = fig.add_subplot(gs[3, 0])
    for c in range(N_CYCLES):
        sp = p[p["scan"] == c]
        axe.plot(sp["v"], sp["i_uA"] / 1000.0, lw=1.0, color=PALETTE[c],
                 label=f"cycle {c+1}")
    axe.axhline(0, color="k", lw=0.5)
    axe.invert_xaxis()
    axe.set(xlabel="potential (V)", ylabel="current (mA)")
    axe.grid(alpha=0.3)
    axe.legend(fontsize=8)
    axe.set_title("Electrochemical CV (i vs E)", fontsize=9.5)

    axq2 = fig.add_subplot(gs[3, 1])
    for c in range(N_CYCLES):
        s = d[d["cycle"] == c].sort_values("t")
        q = s["q_mC"].values - s["q_mC"].values[0]
        y = s["smooth_closed"].values - s["smooth_closed"].values[0]
        axq2.plot(q, y, lw=1.6, color=PALETTE[c], label=f"cycle {c+1}")
    axq2.axhline(0, color="k", lw=0.5)
    axq2.set(xlabel="charge passed within cycle (mC)", ylabel=f"delta {lab.lower()} ({unit})")
    axq2.grid(alpha=0.3)
    axq2.legend(fontsize=8)
    axq2.set_title(f"{lab} vs charge", fontsize=9.5)

    lag, r, r0, curve, lags = align
    axa = fig.add_subplot(gs[3, 2])
    axa.plot(lags, curve, lw=1.4, color="#2b6cb0")
    axa.axvline(0, color="k", ls="--", lw=1, label="t0 = 0 (assumed)")
    axa.axvline(lag, color="r", ls=":", lw=1.2, label=f"best lag {lag:+.0f} s")
    axa.axhline(0, color="k", lw=0.5)
    axa.set(xlabel="scope lag vs potentiostat (s)", ylabel=f"corr({lab.lower()}, charge)")
    axa.grid(alpha=0.3)
    axa.legend(fontsize=7.5)
    axa.set_title(f"Alignment check: r={r0:+.2f} at lag 0", fontsize=9.5)

    fig.suptitle(f"2026-07-29 1000 um ALONG + PPy, new electropol cell - {lab} CV, "
                 f"2.5 mV/s, measured E & i  (L0 {L0_MM} mm, "
                 f"{WIDTH_MM} x {THICKNESS_MM} mm, {AREA_MM2:.3f} mm$^2$)",
                 fontweight="bold")
    out = f"cv_20260729_along_{tag.lower()}_with_current.png"
    fig.savefig(out, bbox_inches="tight", dpi=125)
    print(f"  saved {out}")


plt.rcParams.update({"figure.dpi": 120, "font.size": 9.5})
summary = []
for tag, run in RUNS.items():
    d, p, align = load(tag, run)
    plot_run(tag, run, d, p, align)
    for c in range(N_CYCLES):
        s = d[d["cycle"] == c].sort_values("t")
        y = s["smooth_closed"].values - s["smooth_closed"].values[0]
        swing = y.max() - y.min()
        sp = p[p["scan"] == c]
        summary.append(dict(
            run=tag, cycle=c + 1, unit=run["unit"], swing=round(swing, 4),
            normalised=round(normalise(tag, swing), 4), norm_unit=run["norm_unit"],
            v_at_max=round(s["v_meas"].values[int(np.argmax(y))], 3),
            v_at_min=round(s["v_meas"].values[int(np.argmin(y))], 3),
            anodic_charge_mC=round(np.trapezoid(sp["i_uA"].clip(lower=0), sp["t"]) / 1e3, 2),
            peak_I_mA=round(sp["i_uA"].abs().max() / 1e3, 3)))

print(f"\n=== MEASURED geometry: L0={L0_MM} mm, W={WIDTH_MM} mm, T={THICKNESS_MM} mm, "
      f"area={AREA_MM2:.4f} mm^2 ===")
print("\n=== per-cycle actuation swing (creep-corrected, measured potential) ===")
print(pd.DataFrame(summary).to_string(index=False))

# --- charge balance: a big anodic/cathodic imbalance means irreversible side reactions,
# --- which matters directly for the performance-over-time study.
print("\n=== charge balance per cycle (anodic / cathodic / net, mC) ===")
bal = []
for tag, run in RUNS.items():
    p = load_potentiostat(run["pot"])
    for c in range(N_CYCLES):
        sp = p[p["scan"] == c]
        qa = np.trapezoid(sp["i_uA"].clip(lower=0), sp["t"]) / 1e3
        qc = np.trapezoid(sp["i_uA"].clip(upper=0), sp["t"]) / 1e3
        bal.append(dict(run=tag, cycle=c + 1, anodic_mC=round(qa, 1),
                        cathodic_mC=round(qc, 1), net_mC=round(qa + qc, 1),
                        recovered_pct=round(100 * abs(qc) / qa, 1)))
print(pd.DataFrame(bal).to_string(index=False))
print("recovered_pct = cathodic charge returned as a fraction of anodic charge passed;\n"
      "well under 100% means charge is going somewhere irreversible (side reactions).")
