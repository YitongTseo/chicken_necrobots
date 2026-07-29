"""
Waveform zoom for the 2026-07-29 endurance test: the FIRST 10 and LAST 10 cycles, showing
force, measured current, and applied potential.

TIMEBASE ASSUMPTION (stated because it is load-bearing for the potential overlay):
the scope's elapsed_s comes from the host OS clock, while the potentiostat's step timing
comes from its own firmware timer, and the two disagree by 0.672% (force ripple period
20.1345 s on the scope clock vs 10.000 s steps on the potentiostat clock). We take the
scope/OS clock as the reference, so the potentiostat time axis is rescaled by
PERIOD_SCOPE / PERIOD_POT = 1.00672 and the true drive period is 20.1345 s. Without this,
the two records slip 121 s (~6 cycles) apart by the end of the 5 h run and the last-10-cycle
panel would show a meaningless force-to-drive phase.

The applied potential is NOT logged (the export is MultiStep Amperometry, i vs t only), so
it is reconstructed as a +-1 V square wave with 10 s half-periods - verified against the
current's step transients, which land every 10.000 s of potentiostat time.

A residual ~165 deg of slow phase drift remains after the rescaling and may be real; treat
small phase differences between the two windows with caution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

SCOPE = "1Vto-1V_FORCE_10sec_each_phase_scope_log_20260729_182740.csv"
POT = "potentiostat/07-29-26_FORCE_10sec_1Vto-1V_overtime_test-3.csv"

GAIN_MN_PER_V, NO_LOAD_V = 56.7, 1.25
PERIOD_POT = 20.0
N_WIN = 10                      # cycles per zoom window
N_CYCLES = 893                  # from endurance_20260729_analysis.py

C_FIRST, C_LAST, C_CUR, C_POT = "#2b6cb0", "#c53030", "#b7791f", "0.55"


def load_scope():
    d = pd.read_csv(SCOPE).rename(columns={"elapsed_s": "t"})
    d["force_mN"] = (d["ch2_MEAN_V"] - NO_LOAD_V) * GAIN_MN_PER_V
    return d[d["t"] <= 18000.0].reset_index(drop=True)


def load_pot():
    p = pd.read_csv(StringIO(open(POT, "rb").read().decode("utf-16")), skiprows=5)
    p.columns = ["t", "i_uA"]
    return p.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def refine_period(t, y, lo=19.7, hi=20.3, n=6001):
    ac = y - pd.Series(y).rolling(101, center=True, min_periods=1).mean().values
    best = (np.nan, np.inf)
    for P in np.linspace(lo, hi, n):
        w = 2 * np.pi / P
        X = np.c_[np.cos(w * t), np.sin(w * t)]
        r = ac - X @ np.linalg.lstsq(X, ac, rcond=None)[0]
        ss = float(r @ r)
        if ss < best[1]:
            best = (P, ss)
    return best[0]


def applied_potential(t, period):
    """+1 V for the first half of each period, -1 V for the second."""
    return np.where((np.asarray(t, float) % period) < period / 2, 1.0, -1.0)


d = load_scope()
p = load_pot()
PERIOD = refine_period(d["t"].values, d["force_mN"].values)
SCALE = PERIOD / PERIOD_POT
p["t_corr"] = p["t"] * SCALE                      # potentiostat time on the scope clock

print(f"drive period (scope clock) {PERIOD:.4f} s; potentiostat time rescaled by {SCALE:.5f}")

WINDOWS = [
    ("first", 0, C_FIRST, f"cycles 1-{N_WIN}"),
    ("last", N_CYCLES - N_WIN, C_LAST, f"cycles {N_CYCLES-N_WIN+1}-{N_CYCLES}"),
]

plt.rcParams.update({"figure.dpi": 125, "font.size": 9.5})
fig = plt.figure(figsize=(15.5, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.22)

folded = {}
for col, (tag, k0, colr, lab) in enumerate(WINDOWS):
    t0, t1 = k0 * PERIOD, (k0 + N_WIN) * PERIOD

    # ---------------- row 0: force ----------------------------------------
    m = (d["t"] >= t0) & (d["t"] < t1)
    tt = d["t"].values[m] - t0
    ff = d["force_mN"].values[m]
    ax = fig.add_subplot(gs[0, col])
    ax.plot(tt, ff, lw=1.6, color=colr, marker="o", ms=2.2, mfc="w", mew=0.5)
    ax.set(xlabel="time within window (s)", ylabel="Force (mN)", xlim=(0, t1 - t0))
    ax.grid(alpha=0.3)
    ax.set_title(f"FORCE - {lab}\nmean {ff.mean():.2f} mN, "
                 f"pk-pk {ff.max()-ff.min():.3f} mN", fontsize=10, fontweight="bold",
                 color=colr)
    axp = ax.twinx()
    tg = np.linspace(0, t1 - t0, 4000)
    axp.plot(tg, applied_potential(tg + t0, PERIOD), color=C_POT, lw=1.0, alpha=0.8)
    axp.set_ylabel("applied potential (V)", color=C_POT)
    axp.tick_params(axis="y", colors=C_POT)
    axp.set_ylim(-3.2, 1.4)

    # ---------------- row 1: current --------------------------------------
    mp = (p["t_corr"] >= t0) & (p["t_corr"] < t1)
    tc = p["t_corr"].values[mp] - t0
    ic = p["i_uA"].values[mp] / 1000.0
    axi = fig.add_subplot(gs[1, col])
    axi.plot(tc, ic, lw=1.1, color=C_CUR)
    axi.axhline(0, color="k", lw=0.5)
    axi.set(xlabel="time within window (s)", ylabel="Current (mA)", xlim=(0, t1 - t0))
    axi.grid(alpha=0.3)
    axi.set_title(f"CURRENT - {lab}\npeak |i| {np.abs(ic).max():.2f} mA, "
                  f"anodic {np.trapezoid(np.clip(ic,0,None),tc)/N_WIN:.1f} mC/cycle",
                  fontsize=10)
    axp2 = axi.twinx()
    axp2.plot(tg, applied_potential(tg + t0, PERIOD), color=C_POT, lw=1.0, alpha=0.8)
    axp2.set_ylabel("applied potential (V)", color=C_POT)
    axp2.tick_params(axis="y", colors=C_POT)
    axp2.set_ylim(-1.4, 3.2)

    # keep phase-folded means for row 2
    folded[tag] = dict(
        colour=colr, label=lab,
        f_ph=(d["t"].values[m] % PERIOD), f_y=ff - ff.mean(),
        i_ph=(p["t_corr"].values[mp] % PERIOD), i_y=ic)

# ---------------- row 2: phase-folded overlay -----------------------------
def fold_mean(ph, y, nb=40):
    bins = np.linspace(0, PERIOD, nb + 1)
    idx = np.digitize(ph, bins) - 1
    ctr = 0.5 * (bins[:-1] + bins[1:])
    mu = np.array([y[idx == i].mean() if (idx == i).any() else np.nan for i in range(nb)])
    return ctr, mu


axf = fig.add_subplot(gs[2, 0])
for tag in ("first", "last"):
    g = folded[tag]
    ctr, mu = fold_mean(g["f_ph"], g["f_y"])
    axf.plot(ctr, mu, lw=2.0, marker="o", ms=3, color=g["colour"],
             label=f"{g['label']}  (pk-pk {np.nanmax(mu)-np.nanmin(mu):.3f} mN)")
axf.axvline(PERIOD / 2, color="k", ls="--", lw=1.0)
axf.axhline(0, color="k", lw=0.5)
axf.set(xlabel=f"phase within one {PERIOD:.2f} s cycle (s)",
        ylabel="force - window mean (mN)", xlim=(0, PERIOD))
axf.grid(alpha=0.3)
axf.legend(fontsize=8)
axf.set_title("FORCE folded to one cycle\n(dashed = +1 V -> -1 V switch)", fontsize=10,
              fontweight="bold")

axc = fig.add_subplot(gs[2, 1])
for tag in ("first", "last"):
    g = folded[tag]
    ctr, mu = fold_mean(g["i_ph"], g["i_y"])
    axc.plot(ctr, mu, lw=2.0, color=g["colour"], label=g["label"])
axc.axvline(PERIOD / 2, color="k", ls="--", lw=1.0)
axc.axhline(0, color="k", lw=0.5)
axc.set(xlabel=f"phase within one {PERIOD:.2f} s cycle (s)", ylabel="current (mA)",
        xlim=(0, PERIOD))
axc.grid(alpha=0.3)
axc.legend(fontsize=8)
axc.set_title("CURRENT folded to one cycle", fontsize=10)

fig.suptitle("2026-07-29 endurance test - waveform zoom, first 10 vs last 10 cycles\n"
             f"+1 V / -1 V, 10 s per phase; potentiostat time rescaled x{SCALE:.5f} onto "
             "the scope clock (see script header)", fontweight="bold", fontsize=11.5)
fig.savefig("endurance_20260729_waveform_zoom.png", bbox_inches="tight", dpi=125)
print("saved endurance_20260729_waveform_zoom.png")

for tag in ("first", "last"):
    g = folded[tag]
    _, mf = fold_mean(g["f_ph"], g["f_y"])
    _, mi = fold_mean(g["i_ph"], g["i_y"])
    print(f"{g['label']:22s} force pk-pk {np.nanmax(mf)-np.nanmin(mf):.4f} mN, "
          f"force min at phase {np.nanargmin(mf)*PERIOD/40:.1f} s, "
          f"current pk-pk {np.nanmax(mi)-np.nanmin(mi):.3f} mA")
