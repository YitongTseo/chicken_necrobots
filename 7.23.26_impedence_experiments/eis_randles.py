"""Proper Randles fit: Rs + [ CPE_dl || (Rct + Warburg) ].

Gives the textbook shape - a (depressed) semicircle from Rs-(Rct|CPE) followed by
the ~45deg Warburg tail - which the earlier Rs-(RQ) model could not, because it had
no diffusion element.

Two fixes vs the previous attempt:
  1. TRIM the high-frequency inductive lead-in.  At the top of the sweep the cabling
     adds a small series inductance that hooks -Im(Z) down (sometimes below zero).
     We drop every point at a frequency ABOVE the high-f minimum of -Im(Z) (the knee
     that marks the true Rs intercept).  Pure-capacitive films have their -Im minimum
     at the highest frequency, so nothing is trimmed there.
  2. Add a generalized Warburg  Zw = Aw*(j w)^(-p)  (p=0.5 = ideal 45deg diffusion,
     p->1 = blocking/capacitive), so one model spans both the diffusive constructs
     and the blocking pure-PPy films.

Model admittance of the parallel block:  Y = Qdl*(j w)^ndl + 1/(Rct + Zw)
   Z(w) = Rs + 1/Y
Params: Rs, Qdl, ndl, Rct, Aw, p          (proportional 1/|Z| weighting)
Run:  python3 eis_randles.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from galvani import BioLogic as BL

OUT = "plots"
os.makedirs(OUT, exist_ok=True)
load = lambda n: np.array(BL.MPRfile(n + "_C01.mpr").data)
tstamp = lambda n: str(getattr(BL.MPRfile(n + "_C01.mpr"), "timestamp", ""))[11:19]


def trim_leadin(f, Zre, Znegim):
    """Drop the HF inductive hook: keep points at/below the frequency of the
    high-frequency -Im minimum. Operate on arrays sorted high->low frequency."""
    order = np.argsort(f)[::-1]
    f, Zre, Znegim = f[order], Zre[order], Znegim[order]
    # search the top half of the sweep (by frequency) for the -Im valley
    half = max(3, len(f) // 2)
    knee = int(np.argmin(Znegim[:half]))
    keep = slice(knee, None)
    return f[keep], Zre[keep], Znegim[keep], (f[:knee], Zre[:knee], Znegim[:knee])


def model(p, w):
    Rs, Qdl, ndl, Rct, Aw, pw = p
    jw = 1j * w
    Zw = Aw * jw ** (-pw)
    Y = Qdl * jw ** ndl + 1.0 / (Rct + Zw)
    return Rs + 1.0 / Y


def fit(name, fmin=None):
    d = load(name)
    f0 = d["freq/Hz"]
    Zre0, Znegim0 = d["Re(Z)/Ohm"], d["-Im(Z)/Ohm"]
    f, Zre, Znegim, dropped = trim_leadin(f0, Zre0, Znegim0)
    good = np.isfinite(f)
    if fmin is not None:
        good &= f >= fmin
    f, Zre, Znegim = f[good], Zre[good], Znegim[good]
    Z = Zre - 1j * Znegim
    w = 2 * np.pi * f

    Rs0 = Zre.min()
    span = max(Zre.max() - Rs0, 20.0)
    # arc apex ~ frequency of max -Im in the upper-freq portion
    Rct0 = span * 0.6
    fap = f[np.argmax(Znegim)]
    Qdl0 = 1.0 / (Rct0 * 2 * np.pi * fap)
    # Warburg coeff from the lowest-frequency point
    wlo = 2 * np.pi * f.min()
    Aw0 = max(abs(Z[np.argmin(f)]) - Rs0 - Rct0, 10.0) * wlo ** 0.5

    def resid(lp):
        p = [np.exp(lp[0]), np.exp(lp[1]), lp[2], np.exp(lp[3]), np.exp(lp[4]), lp[5]]
        r = (model(p, w) - Z) / np.abs(Z)
        return np.concatenate([r.real, r.imag])

    best = None
    for rct_mult, aw_mult in [(1, 1), (0.2, 1), (3, 0.3), (1, 5)]:
        lp0 = [np.log(Rs0), np.log(Qdl0), 0.85, np.log(Rct0 * rct_mult),
               np.log(Aw0 * aw_mult), 0.5]
        try:
            res = least_squares(
                resid, lp0, method="trf", max_nfev=30000,
                bounds=([np.log(1), np.log(1e-10), 0.5, np.log(1), np.log(1e-2), 0.3],
                        [np.log(1e6), np.log(1e2), 1.0, np.log(1e9), np.log(1e8), 1.0]))
            if best is None or res.cost < best.cost:
                best = res
        except Exception:
            continue
    lp = best.x
    p = [np.exp(lp[0]), np.exp(lp[1]), lp[2], np.exp(lp[3]), np.exp(lp[4]), lp[5]]
    rmse = np.sqrt(np.mean(best.fun ** 2)) * 100
    return dict(name=name, p=p, Rs=p[0], ndl=p[2], Rct=p[3], pw=p[5],
                rmse=rmse, f=f, Z=Z, dropped=dropped, ts=tstamp(name))


SAMPLES = [
    ("pure_ppy_first_test_0", "pure PPy film A", None),
    ("pure_ppy_B_LiClO4_PC_v2", "pure PPy film B", None),
    ("4dunk_pol_A_LiClO4", "4-dunk (A)", None),
    ("4dunk_pol_B_LiClO4", "4-dunk (B)", None),
    ("2chem_pol_old_method_first_test_LiClO4_PC", "2x chem OLD METHOD", None),
    ("electropol_on_2chem_pol_LiClO4", "electropol on 2x chem", None),
    ("3chem_pol_A_LiClO4", "3x chem pol (A)", None),
    ("2chem_pol_B_LiClO4", "2x chem pol (B)", None),
    ("decell_A_LiClO4", "decell only (0% PPy)", None),
    ("platinum_baseline_LiClO4", "Pt blank", None),
]

results = [dict(fit(n, fm), label=lab) for n, lab, fm in SAMPLES]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, r in zip(axes.ravel(), results):
    df, dre, dim = r["dropped"]
    if len(df):
        ax.plot(dre, dim, "x", ms=5, color="0.7", label="trimmed lead-in", zorder=2)
    ax.plot(r["Z"].real, -r["Z"].imag, "o", ms=4, color="#4c72b0", label="data", zorder=3)
    wf = 2 * np.pi * np.logspace(np.log10(r["f"].min()), np.log10(r["f"].max()), 500)
    Zf = model(r["p"], wf)
    ax.plot(Zf.real, -Zf.imag, "-", color="#c44e52", lw=1.9, label="Randles+W fit", zorder=4)
    ax.set_title(f"{r['label']}  ({r['ts']})", fontsize=9)
    ax.set_xlabel("Re(Z) / Ohm", fontsize=8)
    ax.set_ylabel("-Im(Z) / Ohm", fontsize=8)
    rct = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    ax.text(0.04, 0.96, f"Rs={r['Rs']:.0f}\nRct={rct}\nndl={r['ndl']:.2f}\np_w={r['pw']:.2f}\nfit {r['rmse']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.9))
    ax.legend(fontsize=6.2, loc="lower right")
    ax.grid(alpha=.3)
fig.suptitle("Randles fit  Rs + [CPE || (Rct + Warburg)]  -  LiClO4/PC, HF lead-in trimmed", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT}/9_randles_liclo4.png", dpi=110)
plt.close()

print(f"{'sample':<24}{'time':>9}{'Rs':>7}{'Rct':>9}{'ndl':>6}{'p_w':>6}{'fit%':>7}  trimmed")
for r in results:
    rct = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    print(f"{r['label']:<24}{r['ts']:>9}{r['Rs']:>7.0f}{rct:>9}{r['ndl']:>6.2f}{r['pw']:>6.2f}{r['rmse']:>7.1f}  {len(r['dropped'][0])} pts")
