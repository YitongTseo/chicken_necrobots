"""FeCN counterpart of fig 10: v2 model (Randles + stray Cp) fit to each aqueous
ferri/ferrocyanide spectrum.  Sub-1 Hz points are masked (grey x) - they are
corrupted by clamp-iron contamination (negative -Im, scattered phase) - and the
fit uses only f >= 1 Hz.  Run:  python3 eis_randles_fecn.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from galvani import BioLogic as BL

OUT = "plots"
FMIN = 1.0
os.makedirs(OUT, exist_ok=True)
load = lambda n: np.array(BL.MPRfile(n + "_C01.mpr").data)
tstamp = lambda n: str(getattr(BL.MPRfile(n + "_C01.mpr"), "timestamp", ""))[11:19]


def model(p, w):
    Rs, Qdl, ndl, Rct, Aw, pw, Cp = p
    jw = 1j * w
    Zw = Aw * jw ** (-pw)
    Yr = Qdl * jw ** ndl + 1.0 / (Rct + Zw)
    Zr = Rs + 1.0 / Yr
    return 1.0 / (1j * w * Cp + 1.0 / Zr)


def fit(name):
    d = load(name)
    f_all = d["freq/Hz"]
    re_all, nim_all = d["Re(Z)/Ohm"], d["-Im(Z)/Ohm"]
    keep = np.isfinite(f_all) & (f_all >= FMIN)
    f = f_all[keep]
    Z = re_all[keep] - 1j * nim_all[keep]
    w = 2 * np.pi * f
    dropped = (re_all[~keep & (f_all < FMIN)], nim_all[~keep & (f_all < FMIN)])

    Rs0 = re_all[np.argmax(f_all)]
    span = max(re_all.max() - Rs0, 20.0)
    Rct0 = span * 0.6
    fap = f[np.argmax(nim_all[keep])]
    Qdl0 = 1.0 / (Rct0 * 2 * np.pi * fap)
    Aw0 = max(abs(Z[np.argmin(f)]) - Rs0 - Rct0, 10.0) * (2 * np.pi * f.min()) ** 0.5
    Zhi = Z[np.argmax(f)]
    Cp0 = max((-Zhi.imag) / (2 * np.pi * f.max() * abs(Zhi) ** 2), 1e-11)

    def resid(lp):
        p = [np.exp(lp[0]), np.exp(lp[1]), lp[2], np.exp(lp[3]), np.exp(lp[4]), lp[5], np.exp(lp[6])]
        r = (model(p, w) - Z) / np.abs(Z)
        return np.concatenate([r.real, r.imag])

    lo = [np.log(1), np.log(1e-10), 0.5, np.log(1), np.log(1e-2), 0.3, np.log(1e-12)]
    hi = [np.log(1e6), np.log(1e2), 1.0, np.log(1e9), np.log(1e8), 1.0, np.log(1e-6)]
    best = None
    for rm, am, cm in [(1, 1, 1), (0.3, 1, 1), (2, 0.3, 1), (1, 3, 0.3), (0.5, 1, 3)]:
        lp0 = [np.log(Rs0 * 0.9), np.log(Qdl0), 0.85, np.log(Rct0 * rm), np.log(Aw0 * am), 0.5, np.log(Cp0 * cm)]
        lp0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(lp0)]
        try:
            res = least_squares(resid, lp0, method="trf", max_nfev=40000, bounds=(lo, hi))
            if best is None or res.cost < best.cost:
                best = res
        except Exception:
            continue
    lp = best.x
    p = [np.exp(lp[0]), np.exp(lp[1]), lp[2], np.exp(lp[3]), np.exp(lp[4]), lp[5], np.exp(lp[6])]
    rmse = np.sqrt(np.mean(best.fun ** 2)) * 100
    return dict(name=name, p=p, Rs=p[0], Rct=p[3], ndl=p[2], pw=p[5], Cp=p[6],
                rmse=rmse, f=f, Z=Z, dropped=dropped, ts=tstamp(name))


SAMPLES = [
    ("pure_ppy_fourth_test_FeCN", "pure PPy film A"),
    ("pure_ppy_B_FeCN", "pure PPy film B"),
    ("4dunk_pol_A_FeCN", "4-dunk (A)"),
    ("4dunk_pol_B_FeCN", "4-dunk (B)"),
    ("2chem_pol_old_method_second_test_FeCN", "2x chem pol (A)"),
    ("electropol_on_2chem_pol_FeCN", "electropol on 2x chem"),
    ("3chem_pol_A_FeCN", "3x chem pol (A)"),
    ("2chem_pol_B_FeCN", "2x chem pol (B)"),
    ("decell_A_LCN", "decell only (0% PPy)"),
    ("platinum_baseline_FeCN", "Pt blank"),
]
results = [dict(fit(n), label=lab) for n, lab in SAMPLES]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, r in zip(axes.ravel(), results):
    dre, dim = r["dropped"]
    if len(dre):
        ax.plot(dre, dim, "x", ms=5, color="0.7", label="masked (<1 Hz)", zorder=2)
    ax.plot(r["Z"].real, -r["Z"].imag, "o", ms=4, color="#e07b39", label="data (fit)", zorder=3)
    wf = 2 * np.pi * np.logspace(np.log10(r["f"].min()), np.log10(r["f"].max()), 600)
    Zf = model(r["p"], wf)
    ax.plot(Zf.real, -Zf.imag, "-", color="#8c2d04", lw=1.9, label="fit (+stray C)", zorder=4)
    ax.set_title(f"{r['label']}  ({r['ts']})", fontsize=9)
    ax.set_xlabel("Re(Z) / Ohm", fontsize=8)
    ax.set_ylabel("-Im(Z) / Ohm", fontsize=8)
    rct = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    ax.text(0.04, 0.96, f"Rs={r['Rs']:.0f}\nRct={rct}\nndl={r['ndl']:.2f}\np_w={r['pw']:.2f}\nCp={r['Cp']*1e9:.1f}nF\nfit {r['rmse']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.9))
    ax.legend(fontsize=6.0, loc="lower right")
    ax.grid(alpha=.3)
fig.suptitle("Randles + parasitic parallel C  -  aq. FeCN (+0.23 V vs Ref; sub-1 Hz masked)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT}/13_randles_fecn.png", dpi=110)
plt.close()

print(f"{'sample':<24}{'time':>9}{'Rs':>7}{'Rct':>9}{'ndl':>6}{'p_w':>6}{'Cp/nF':>7}{'fit%':>7}")
for r in results:
    rct = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    print(f"{r['label']:<24}{r['ts']:>9}{r['Rs']:>7.0f}{rct:>9}{r['ndl']:>6.2f}{r['pw']:>6.2f}{r['Cp']*1e9:>7.1f}{r['rmse']:>7.1f}")
