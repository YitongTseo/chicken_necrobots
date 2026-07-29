"""
Endurance / performance-over-time test, 2026-07-29.
+1 V for 10 s, -1 V for 10 s (20 s period), 5 hours = 900 cycles.

  scope       1Vto-1V_FORCE_10sec_each_phase_scope_log_20260729_182740.csv
              CH2 x 56.7 mN/V, no-load baseline 1.25 V. Runs 0 -> 18000.4 s = exactly
              900.0 cycles, then stops (the "cut out" after 5 h).
  potentiostat potentiostat/07-29-26_FORCE_10sec_1Vto-1V_overtime_test-3.csv
              MultiStep Amperometry, i vs t, 0.1 s resolution, 0 -> 20000 s.

The potentiostat outlasts the scope by 2000 s, so all joint analysis is truncated to the
scope's 18000 s / 900 cycles, as the user suggested. Drive timing verified from the current
itself: step transients every 10.0 s, so period = 20 s with t=0 starting a +1 V phase.

Two questions:
  1. CREEP - how does the resting (baseline) force drift over 5 h, and what law fits it?
  2. OUTPUT - how much actuation amplitude is left at cycle 900, and does the ELECTRICAL
     drive decay with it? That distinguishes material fatigue from electrochemical decay -
     the question the force channel alone could not answer on the earlier short runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.optimize import curve_fit

SCOPE = "1Vto-1V_FORCE_10sec_each_phase_scope_log_20260729_182740.csv"
POT = "potentiostat/07-29-26_FORCE_10sec_1Vto-1V_overtime_test-3.csv"

GAIN_MN_PER_V = 56.7
NO_LOAD_V = 1.25

# The two instruments do NOT agree on how long a cycle is. The potentiostat steps every
# 10.000 s on its own clock (verified from its current transients), but the force
# oscillation has a period of ~20.134 s on the scope's clock - a stable 0.67% timebase
# mismatch (20.118 s fitted on the first half, 20.142 s on the second), which accumulates
# to ~121 s over 5 h. Left uncorrected it shows up as a spurious 2040 deg of "phase drift".
# Each instrument is therefore binned into cycles using its OWN period. This does not
# affect any trend below: baseline and amplitude are per-cycle quantities that vary
# smoothly over hundreds of cycles, so a <=6-cycle registration offset at the end of the
# run is immaterial.
PERIOD_POT = 20.0                  # potentiostat clock
PERIOD_SCOPE = None                # measured at run time from the force trace
SCOPE_SPAN_S = 18000.0             # scope stops here ("cut out" after 5 h)

# Geometry measured 2026-07-30 (same sample as the CV runs)
L0_MM, WIDTH_MM, THICKNESS_MM = 4.0, 5.0, 0.2230
AREA_MM2 = WIDTH_MM * THICKNESS_MM

C_FORCE, C_CUR, C_FIT = "#c53030", "#b7791f", "#2b6cb0"


# ----------------------------------------------------------------- loading
def load_scope():
    d = pd.read_csv(SCOPE).rename(columns={"elapsed_s": "t"})
    d["force_mN"] = (d["ch2_MEAN_V"] - NO_LOAD_V) * GAIN_MN_PER_V
    return d[d["t"] <= SCOPE_SPAN_S].reset_index(drop=True)


def load_pot():
    txt = open(POT, "rb").read().decode("utf-16")
    p = pd.read_csv(StringIO(txt), skiprows=5)
    p.columns = ["t", "i_uA"]
    p = p.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return p


# ------------------------------------------------- timebase / period measurement
def ac_part(t, y, win=101):
    """Actuation ripple with the slow creep removed."""
    return y - pd.Series(y).rolling(win, center=True, min_periods=1).mean().values


def refine_period(t, ac, lo=19.7, hi=20.3, n=6001):
    """Global least-squares period of the force ripple on the SCOPE timebase."""
    Ps = np.linspace(lo, hi, n)
    best = (np.nan, np.inf)
    for P in Ps:
        w = 2 * np.pi / P
        X = np.c_[np.cos(w * t), np.sin(w * t)]
        r = ac - X @ np.linalg.lstsq(X, ac, rcond=None)[0]
        ss = float(r @ r)
        if ss < best[1]:
            best = (P, ss)
    return best[0]


def phase_drift(t, ac, P, n_cycles):
    """Cycle-by-cycle phase at period P; a steady ramp means P is wrong."""
    w = 2 * np.pi / P
    ph = []
    for k in range(n_cycles):
        m = (t >= k * P) & (t < (k + 1) * P)
        if m.sum() < 10:
            ph.append(np.nan)
            continue
        tt, aa = t[m], ac[m]
        a, b = np.linalg.lstsq(np.c_[np.cos(w * tt), np.sin(w * tt)], aa, rcond=None)[0]
        ph.append(np.arctan2(b, a))
    ph = np.array(ph, float)
    ok = np.isfinite(ph)
    out = np.full(len(ph), np.nan)
    out[ok] = np.degrees(np.unwrap(ph[ok]))
    return out


# ------------------------------------------------------------ per-cycle metrics
def per_cycle(d, p, p_scope, n_cycles):
    w = 2 * np.pi / p_scope
    rows = []
    ti, ii = p["t"].values, p["i_uA"].values
    for k in range(n_cycles):
        t0, t1 = k * p_scope, (k + 1) * p_scope
        m = (d["t"].values >= t0) & (d["t"].values < t1)
        tt = d["t"].values[m]
        ff = d["force_mN"].values[m]
        if len(tt) < 10:
            continue
        # fundamental amplitude by sine fit (robust at ~41 samples/cycle)
        X = np.c_[np.cos(w * tt), np.sin(w * tt), np.ones_like(tt)]
        a, b, c0 = np.linalg.lstsq(X, ff, rcond=None)[0]
        # potentiostat binned on ITS own clock
        q0, q1 = k * PERIOD_POT, (k + 1) * PERIOD_POT
        mp = (ti >= q0) & (ti < q1)
        tc, ic = ti[mp], ii[mp]
        qa = np.trapezoid(np.clip(ic, 0, None), tc) / 1e3 if len(tc) > 2 else np.nan
        qc = np.trapezoid(np.clip(ic, None, 0), tc) / 1e3 if len(tc) > 2 else np.nan
        rows.append(dict(
            cycle=k + 1, t_mid=0.5 * (t0 + t1),
            baseline_mN=ff.mean(),
            amp_mN=2 * np.hypot(a, b),
            p2p_mN=ff.max() - ff.min(),
            anodic_mC=qa, cathodic_mC=qc, net_mC=qa + qc,
            peak_i_mA=np.abs(ic).max() / 1e3 if len(tc) else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ creep models
def m_lin(t, a, b):
    return a + b * t


def m_log(t, a, b):
    return a + b * np.log1p(t / 60.0)


def m_pow(t, a, b, n):
    return a + b * np.power(t / 60.0, n)


def m_exp(t, a, b, tau):
    return a + b * (1.0 - np.exp(-t / tau))


CREEP_MODELS = [
    ("linear            a+b*t", m_lin, [160, 1e-3]),
    ("logarithmic  a+b*ln(1+t)", m_log, [160, 1.0]),
    ("power law     a+b*t^n", m_pow, [160, 1.0, 0.5]),
    ("exp. saturating", m_exp, [160, 15.0, 3000.0]),
]


def fit_models(t, y, models):
    out = []
    for name, fn, p0 in models:
        try:
            pars, _ = curve_fit(fn, t, y, p0=p0, maxfev=40000)
            resid = y - fn(t, *pars)
            r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
            out.append((name, fn, pars, r2, np.sqrt((resid ** 2).mean())))
        except Exception as e:                                   # noqa: BLE001
            out.append((name, fn, None, -np.inf, np.nan))
    return sorted(out, key=lambda r: -r[3])


# ============================================================== run
d = load_scope()
p = load_pot()
ac = ac_part(d["t"].values, d["force_mN"].values)
PERIOD_SCOPE = refine_period(d["t"].values, ac)
N_CYCLES = int(min(d["t"].max() // PERIOD_SCOPE, p["t"].max() // PERIOD_POT))
TOTAL_S = N_CYCLES * PERIOD_SCOPE

print(f"scope: {len(d)} samples, 0 -> {d['t'].max():.1f} s, "
      f"median dt {np.median(np.diff(d['t'])):.3f} s")
print(f"potentiostat: {len(p)} samples, 0 -> {p['t'].max():.0f} s "
      f"(outlasts the scope by {p['t'].max()-d['t'].max():.0f} s, discarded)")
print(f"absolute force {d['force_mN'].min():.1f} - {d['force_mN'].max():.1f} mN")

print(f"\n=== TIMEBASE ===")
print(f"  potentiostat period (its own clock): {PERIOD_POT:.4f} s")
print(f"  force-ripple period (scope clock):   {PERIOD_SCOPE:.4f} s"
      f"   -> {100*(PERIOD_SCOPE-PERIOD_POT)/PERIOD_POT:+.3f}% mismatch, "
      f"{d['t'].max()*(PERIOD_SCOPE-PERIOD_POT)/PERIOD_POT:+.0f} s over the run")
ph20 = phase_drift(d["t"].values, ac, PERIOD_POT, N_CYCLES)
phb = phase_drift(d["t"].values, ac, PERIOD_SCOPE, N_CYCLES)
print(f"  cycle-to-cycle phase drift at 20.000 s: {np.nanmax(ph20)-np.nanmin(ph20):.0f} deg"
      f"  (spurious - this is the mismatch, not degradation)")
print(f"  cycle-to-cycle phase drift at {PERIOD_SCOPE:.4f} s: "
      f"{np.nanmax(phb)-np.nanmin(phb):.0f} deg  (residual)")
print(f"  analysing {N_CYCLES} cycles")

# (removed: the force-vs-charge lag test is uninformative at a 10 s drive - |r| stayed
# under 0.09 at every lag because the mechanical response is so heavily attenuated. It is
# also unnecessary: per-cycle baseline (a mean) and amplitude (a sine-fit magnitude) are
# both phase-invariant, so the numbers below do not depend on sub-period alignment.)

c = per_cycle(d, p, PERIOD_SCOPE, N_CYCLES)
print(f"per-cycle table: {len(c)} cycles")

# ---- creep ------------------------------------------------------------------
fits = fit_models(c["t_mid"].values, c["baseline_mN"].values, CREEP_MODELS)
print("\n=== CREEP: baseline force vs time, model comparison ===")
for name, fn, pars, r2, rmse in fits:
    ps = "fit failed" if pars is None else np.array2string(pars, precision=4)
    print(f"  {name:26s} R2={r2:7.4f}  RMSE={rmse:.3f} mN   {ps}")
best_name, best_fn, best_pars, best_r2, _ = fits[0]

b0, b1 = c["baseline_mN"].iloc[0], c["baseline_mN"].iloc[-1]
print(f"\n  baseline {b0:.2f} -> {b1:.2f} mN over 5 h  ({b1-b0:+.2f} mN, "
      f"{100*(b1-b0)/b0:+.1f}%)")
half = c[c["cycle"] <= N_CYCLES // 2]
print(f"  first half drift {half['baseline_mN'].iloc[-1]-b0:+.2f} mN, "
      f"second half {b1-half['baseline_mN'].iloc[-1]:+.2f} mN "
      "(equal halves would mean linear creep)")

# ---- output over time -------------------------------------------------------
def early_late(col, n=25):
    e = c[col].head(n).mean()
    l = c[col].tail(n).mean()
    return e, l, 100 * l / e


print(f"\n=== OUTPUT over {N_CYCLES} cycles (mean of first 25 vs last 25 cycles) ===")
for col, unit in [("amp_mN", "mN"), ("p2p_mN", "mN"), ("anodic_mC", "mC"),
                  ("cathodic_mC", "mC"), ("peak_i_mA", "mA")]:
    e, l, pct = early_late(col)
    print(f"  {col:12s} {e:9.3f} -> {l:9.3f} {unit:3s}   {pct:6.1f}% retained")

amp_e, amp_l, amp_pct = early_late("amp_mN")
qa_e, qa_l, qa_pct = early_late("anodic_mC")
print(f"\n  mechanical amplitude retained {amp_pct:.1f}%, anodic charge retained "
      f"{qa_pct:.1f}%")
print("  -> mechanical decay much faster than charge decay implies material fatigue /\n"
      "     mechanical decoupling; both decaying together implies electrochemical loss.")

# efficiency: mN of swing per mC of anodic charge
c["mN_per_mC"] = c["amp_mN"] / c["anodic_mC"]
ee, el, epct = early_late("mN_per_mC")
print(f"\n  actuation efficiency {ee:.4f} -> {el:.4f} mN swing per mC anodic "
      f"({epct:.1f}% retained)")

afits = fit_models(c["cycle"].values.astype(float), c["amp_mN"].values,
                   [("exp. decay to floor", lambda x, a, b, tau: a + b * np.exp(-x / tau),
                     [0.5, 2.0, 200.0]),
                    ("power law   a*x^-n", lambda x, a, n: a * np.power(x, -n),
                     [3.0, 0.3]),
                    ("linear", m_lin, [2.0, -1e-3])])
print("\n=== amplitude decay, model comparison ===")
for name, fn, pars, r2, rmse in afits:
    ps = "fit failed" if pars is None else np.array2string(pars, precision=4)
    print(f"  {name:22s} R2={r2:7.4f}  RMSE={rmse:.4f} mN   {ps}")

c.to_csv("endurance_20260729_per_cycle.csv", index=False)
print("\nsaved endurance_20260729_per_cycle.csv")

# ============================================================== FIGURE
plt.rcParams.update({"figure.dpi": 125, "font.size": 9.5})
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.28)

# --- row 1: full force trace + per-cycle baseline ---------------------------
ax = fig.add_subplot(gs[0, :])
ax.plot(d["t"] / 3600, d["force_mN"], lw=0.35, color="0.72", label="force (all samples)")
ax.plot(c["t_mid"] / 3600, c["baseline_mN"], lw=2.0, color=C_FORCE,
        label="per-cycle mean (creep)")
ax.set(xlabel="time (h)", ylabel="Force (mN)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
ax.set_title(f"Endurance test: force over 5 h / {N_CYCLES} cycles at +-1 V, 10 s per phase",
             fontsize=10.5, fontweight="bold")

# --- row 2: current envelope + charge per cycle -----------------------------
ax = fig.add_subplot(gs[1, :])
ax.plot(c["t_mid"] / 3600, c["peak_i_mA"], lw=1.4, color=C_CUR,
        label="peak |i| per cycle")
ax.set(xlabel="time (h)", ylabel="peak |current| (mA)")
ax.grid(alpha=0.3)
axq = ax.twinx()
axq.plot(c["t_mid"] / 3600, c["anodic_mC"], lw=1.6, color="#6b46c1",
         label="anodic charge")
axq.plot(c["t_mid"] / 3600, -c["cathodic_mC"], lw=1.6, ls="--", color="#38a169",
         label="|cathodic charge|")
axq.set_ylabel("charge per cycle (mC)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = axq.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper right")
ax.set_title(f"Electrical drive over the same {N_CYCLES} cycles", fontsize=10.5)

# --- row 3 col 0-1: creep with best-fit law --------------------------------
ax = fig.add_subplot(gs[2, :2])
ax.plot(c["t_mid"] / 3600, c["baseline_mN"], lw=1.8, color=C_FORCE, label="per-cycle mean")
if best_pars is not None:
    tt = np.linspace(c["t_mid"].min(), c["t_mid"].max(), 500)
    ax.plot(tt / 3600, best_fn(tt, *best_pars), lw=1.6, ls="--", color=C_FIT,
            label=f"best fit: {best_name.split()[0]} (R$^2$={best_r2:.3f})")
ax.set(xlabel="time (h)", ylabel="baseline force (mN)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title(f"CREEP: baseline {b0:.1f} -> {b1:.1f} mN ({b1-b0:+.1f} mN, "
             f"{100*(b1-b0)/b0:+.1f}%)", fontsize=10.5, fontweight="bold")

# --- row 3 col 2: creep rate ------------------------------------------------
ax = fig.add_subplot(gs[2, 2])
rate = np.gradient(c["baseline_mN"].values,
                   c["t_mid"].values) * 3600            # mN/h
ax.plot(c["t_mid"] / 3600, pd.Series(rate).rolling(25, center=True,
                                                   min_periods=1).mean(),
        lw=1.5, color=C_FIT)
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="time (h)", ylabel="creep rate (mN/h)")
ax.grid(alpha=0.3)
ax.set_title("Creep rate (25-cycle smooth)", fontsize=10)

# --- row 4 col 0: amplitude decay, mechanical vs electrical ----------------
ax = fig.add_subplot(gs[3, 0])
sm = lambda s: pd.Series(s).rolling(15, center=True, min_periods=1).mean()
ax.plot(c["cycle"], 100 * sm(c["amp_mN"]) / c["amp_mN"].head(25).mean(), lw=1.6,
        color=C_FORCE, label="force amplitude")
ax.plot(c["cycle"], 100 * sm(c["anodic_mC"]) / c["anodic_mC"].head(25).mean(), lw=1.6,
        color="#6b46c1", label="anodic charge")
ax.axhline(100, color="k", lw=0.5, ls=":")
ax.set(xlabel="cycle", ylabel="% of first-25-cycle mean")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title(f"Retention at cycle {N_CYCLES}\nforce {amp_pct:.0f}%, charge {qa_pct:.0f}%",
             fontsize=10, fontweight="bold")

# --- row 4 col 1: first vs last cycles, raw --------------------------------
ax = fig.add_subplot(gs[3, 1])
for lab, k0, colr in [(f"cycles 1-3", 0, "#2b6cb0"),
                      (f"cycles {N_CYCLES-2}-{N_CYCLES}", N_CYCLES-3, "#c53030")]:
    m = (d["t"] >= k0 * PERIOD_SCOPE) & (d["t"] < (k0 + 3) * PERIOD_SCOPE)
    tt = d["t"].values[m] - k0 * PERIOD_SCOPE
    ff = d["force_mN"].values[m]
    ax.plot(tt, ff - ff.mean(), lw=1.6, color=colr, label=lab)
for b in np.arange(PERIOD_SCOPE/2, 3*PERIOD_SCOPE, PERIOD_SCOPE/2):
    ax.axvline(b, color="k", ls=":", lw=0.6, alpha=0.4)
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="time within window (s)", ylabel="force - mean (mN)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Waveform, start vs end\n(dotted = 10 s phase boundaries)", fontsize=10)

# --- row 4 col 2: efficiency -----------------------------------------------
ax = fig.add_subplot(gs[3, 2])
ax.plot(c["cycle"], sm(c["mN_per_mC"]), lw=1.6, color="#dd6b20")
ax.set(xlabel="cycle", ylabel="mN swing per mC anodic")
ax.grid(alpha=0.3)
ax.set_title(f"Actuation efficiency\n{ee:.3f} -> {el:.3f} ({epct:.0f}% retained)",
             fontsize=10)

fig.suptitle("2026-07-29 endurance test - 1000 um ALONG + PPy, new electropol cell\n"
             f"+1 V / -1 V, 10 s per phase, {N_CYCLES} cycles over 5 h; "
             f"L0 {L0_MM} mm, {WIDTH_MM} x {THICKNESS_MM} mm ({AREA_MM2:.3f} mm$^2$)",
             fontweight="bold", fontsize=12)
fig.savefig("endurance_20260729_analysis.png", bbox_inches="tight", dpi=125)
print("saved endurance_20260729_analysis.png")
