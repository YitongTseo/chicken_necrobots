"""
Diagnostics for the 2026-07-29 square-wave step runs (+0.6 V / -0.6 V, 10 s per phase).

The two runs looked "wonky", so this script asks three questions:

  1. Is the drive even getting through?  -> lock-in at the 20 s drive period.
  2. How big is the response vs the CV run? -> amplitude in mN, against the 22 mN CV swing.
  3. Can the sample follow a 10 s step?     -> harmonic content of the response.

Answers (see printout): the drive IS locked in at 20.08 s, but the response is ~0.4 mN
pk-pk (about 2% of the CV swing) and almost harmonic-free, i.e. the sample is acting as
a heavy low-pass on the square drive. The run-to-run amplitude envelope collapses and
recovers WITHOUT a phase flip, so it is not a sign reversal.

FORCE run -> CH2, 56.7 mN/V, no-load baseline 1.25 V (so absolute force = (V-1.25)*gain).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

GAIN_MN_PER_V = 56.7
NO_LOAD_V = 1.25            # scope volts at zero load, from the 07.20 pretension sweep
NOMINAL_PERIOD_S = 20.0     # 10 s at +0.6 V + 10 s at -0.6 V
CV_SWING_MN = 22.9          # cycle-3 swing from the same sample's 2.5 mV/s CV, for scale

RUNS = [
    ("whacky_0.6V-neg0.6V_FORCE_10sec_each_phase_scope_log_20260729_175832.csv",
     "run 1 (17:58)"),
    ("whacky2_0.6V-neg0.6V_FORCE_10sec_each_phase_scope_log_20260729_180802.csv",
     "run 2 (18:08)"),
]


def refine_period(t, y, lo=19.0, hi=21.0, n=801):
    """Least-squares best-fit drive period - confirms the drive rather than assuming it."""
    best = (np.nan, np.inf)
    for P in np.linspace(lo, hi, n):
        w = 2 * np.pi / P
        X = np.c_[np.cos(w * t), np.sin(w * t)]
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        ss = float(r @ r)
        if ss < best[1]:
            best = (P, ss)
    return best[0]


def lockin(t, y, P, n_periods=3):
    """Sliding-window amplitude and phase at period P."""
    w = 2 * np.pi / P
    W, step = n_periods * P, P / 4
    ctr = np.arange(t[0] + W / 2, t[-1] - W / 2, step)
    amp, pha = [], []
    for c in ctr:
        m = (t > c - W / 2) & (t < c + W / 2)
        if m.sum() < 20:
            amp.append(np.nan)
            pha.append(np.nan)
            continue
        tt, yy = t[m], y[m] - y[m].mean()
        a, b = np.linalg.lstsq(np.c_[np.cos(w * tt), np.sin(w * tt)], yy, rcond=None)[0]
        amp.append(2 * np.hypot(a, b))
        pha.append(np.arctan2(b, a))
    return ctr, np.array(amp), np.degrees(np.unwrap(np.array(pha)))


def harmonics(t, y, P, orders=(1, 3, 5)):
    w = 2 * np.pi / P
    out = {}
    for h in orders:
        a, b = np.linalg.lstsq(np.c_[np.cos(h * w * t), np.sin(h * w * t)], y,
                               rcond=None)[0]
        out[h] = 2 * np.hypot(a, b)
    return out


plt.rcParams.update({"figure.dpi": 120, "font.size": 9.5})
fig, axs = plt.subplots(3, 2, figsize=(15, 11))

for j, (path, lab) in enumerate(RUNS):
    d = pd.read_csv(path)
    t = d["elapsed_s"].values
    force = (d["ch2_MEAN_V"].values - NO_LOAD_V) * GAIN_MN_PER_V

    # uniform grid, then split into slow baseline (creep) + fast residual (actuation)
    tg = np.arange(t[0], t[-1], 0.5)
    yg = np.interp(tg, t, force)
    base = savgol_filter(yg, int(45 / 0.5) | 1, 2)      # 45 s > 2 drive periods
    res = yg - base
    drift = np.gradient(base, tg) * 60                  # mN/min

    P = refine_period(tg, res)
    ctr, amp, pha = lockin(tg, res, P)
    H = harmonics(tg, res, P)
    i_min = int(np.nanargmin(amp))

    ok = np.isfinite(amp)
    r, p = pearsonr(amp[ok], np.interp(ctr, tg, drift)[ok])
    rs, ps = spearmanr(amp[ok], np.interp(ctr, tg, drift)[ok])

    print(f"--- {lab} ---")
    print(f"  duration {t[-1]:.0f} s, {len(t)} samples, median dt {np.median(np.diff(t)):.3f} s")
    print(f"  absolute force (pretension) {force.min():.1f} - {force.max():.1f} mN")
    print(f"  best-fit drive period {P:.3f} s (nominal {NOMINAL_PERIOD_S})")
    print(f"  fundamental {H[1]:.3f} mN pk-pk = {100*H[1]/CV_SWING_MN:.1f}% of the "
          f"{CV_SWING_MN} mN CV swing")
    print(f"  3rd harmonic {H[3]:.4f} mN = {100*H[3]/H[1]:.1f}% of fundamental "
          f"(a step-following system would show ~33%)")
    print(f"  5th harmonic {H[5]:.4f} mN = {100*H[5]/H[1]:.1f}% of fundamental")
    print(f"  amplitude envelope: {amp[0]:.3f} -> min {amp[i_min]:.3f} mN at "
          f"t={ctr[i_min]:.0f} s -> {amp[-1]:.3f} mN")
    print(f"  phase across the null: {pha[i_min]-pha[0]:+.0f} deg before, "
          f"{pha[-1]-pha[i_min]:+.0f} deg after (a sign reversal would be +-180)")
    print(f"  amplitude vs creep rate: pearson r={r:+.3f} (p={p:.1e}), "
          f"spearman={rs:+.3f} (p={ps:.1e}) -> not the explanation")
    print(f"  baseline creep {base[-1]-base[0]:+.2f} mN over the run\n")

    ax = axs[0, j]
    ax.plot(t, force, lw=0.7, color="tab:blue")
    ax.plot(tg, base, lw=1.4, color="k", alpha=0.6, label="baseline (45 s smooth)")
    ax.set(xlabel="elapsed time (s)", ylabel="Force (mN)",
           title=f"{lab}: raw force  (pretension ~{np.median(force):.0f} mN)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axs[1, j]
    ax.plot(tg, res, lw=0.8, color="0.35")
    ax.set(xlabel="elapsed time (s)", ylabel="Force - baseline (mN)", xlim=(0, 220),
           title=f"{lab}: actuation residual, first 220 s")
    for b in np.arange(0, 220, P):
        ax.axvline(b, color="r", alpha=0.2, lw=0.7)
    ax.grid(alpha=0.3)

    ax = axs[2, j]
    ax.plot(ctr, amp, "o-", ms=3, color="tab:blue")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="elapsed time (s)", ylabel="amplitude (mN pk-pk)", ylim=(0, None),
           title=f"{lab}: lock-in amplitude @ {P:.2f} s (blue) & phase (red)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ctr, pha - pha[0], lw=1.2, color="tab:red")
    ax2.set_ylabel("phase rel. to start (deg)", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    ax2.set_ylim(-200, 200)
    ax2.axhline(180, ls="--", lw=0.7, color="tab:red", alpha=0.5)
    ax2.axhline(-180, ls="--", lw=0.7, color="tab:red", alpha=0.5)

fig.suptitle("2026-07-29 square-wave steps (+0.6 V / -0.6 V, 10 s each): why the "
             "response looks wonky", fontweight="bold")
fig.tight_layout()
fig.savefig("square_wave_20260729_diagnostics.png", bbox_inches="tight", dpi=130)
print("saved square_wave_20260729_diagnostics.png")
