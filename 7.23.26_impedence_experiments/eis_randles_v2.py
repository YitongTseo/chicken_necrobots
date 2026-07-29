"""Randles fit WITH a parasitic parallel capacitance across the cell.

Why v2: the high-impedance samples show a real descending arc at the TOP of the
sweep (100 kHz, -Im ~ 400 Ohm falling to a knee). v1 wrongly trimmed it as an
inductive lead-in and put Rs at the knee -> wrecked fits. That arc is stray
capacitance (cable + cell + reference): it appears only in high-|Z| samples, scales
with |Z|, and is near-identical for two different ~2 kOhm samples. So we model it.

Model:  Z(w) = 1 / ( j w Cp + 1 / Zr )          # Cp = stray capacitance across cell
        Zr   = Rs + 1 / ( Qdl (j w)^ndl + 1/(Rct + Aw (j w)^-pw) )   # Randles + Warburg
Params: Rs, Qdl, ndl, Rct, Aw, pw, Cp   (7)   proportional 1/|Z| weighting, NO trimming.
Run:  python3 eis_randles_v2.py
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


def model(p, w):
    Rs, Qdl, ndl, Rct, Aw, pw, Cp = p
    jw = 1j * w
    Zw = Aw * jw ** (-pw)
    Yr = Qdl * jw ** ndl + 1.0 / (Rct + Zw)
    Zr = Rs + 1.0 / Yr
    return 1.0 / (1j * w * Cp + 1.0 / Zr)


def fit(name):
    d = load(name)
    f = d["freq/Hz"]
    Znegim = d["-Im(Z)/Ohm"]
    good = Znegim > -abs(d["|Z|/Ohm"]).max()          # keep everything real; only drop hard NaNs
    good &= np.isfinite(f)
    f = f[good]
    Z = d["Re(Z)/Ohm"][good] - 1j * Znegim[good]
    w = 2 * np.pi * f

    Rs0 = d["Re(Z)/Ohm"][np.argmax(d["freq/Hz"])]      # HF real intercept (upper bound on Rs)
    span = max(d["Re(Z)/Ohm"].max() - Rs0, 20.0)
    Rct0 = span * 0.6
    fap = f[np.argmax(Znegim[good])]
    Qdl0 = 1.0 / (Rct0 * 2 * np.pi * fap)
    wlo = 2 * np.pi * f.min()
    Aw0 = max(abs(Z[np.argmin(f)]) - Rs0 - Rct0, 10.0) * wlo ** 0.5
    # stray C from the HF point: -Im ~ w Cp |Z|^2
    fhi = f.max()
    Zhi = Z[np.argmax(f)]
    Cp0 = max((-Zhi.imag) / (2 * np.pi * fhi * abs(Zhi) ** 2), 1e-11)

    def resid(lp):
        p = [np.exp(lp[0]), np.exp(lp[1]), lp[2], np.exp(lp[3]),
             np.exp(lp[4]), lp[5], np.exp(lp[6])]
        r = (model(p, w) - Z) / np.abs(Z)
        return np.concatenate([r.real, r.imag])

    lo = [np.log(1), np.log(1e-10), 0.5, np.log(1), np.log(1e-2), 0.3, np.log(1e-12)]
    hi = [np.log(1e6), np.log(1e2), 1.0, np.log(1e9), np.log(1e8), 1.0, np.log(1e-6)]
    best = None
    for rm, am, cm in [(1, 1, 1), (0.3, 1, 1), (2, 0.3, 1), (1, 3, 0.3), (0.5, 1, 3)]:
        lp0 = [np.log(Rs0 * 0.9), np.log(Qdl0), 0.85, np.log(Rct0 * rm),
               np.log(Aw0 * am), 0.5, np.log(Cp0 * cm)]
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
                rmse=rmse, f=f, Z=Z, ts=tstamp(name))


SAMPLES = [
    ("pure_ppy_first_test_0", "pure PPy film A"),
    ("pure_ppy_B_LiClO4_PC_v2", "pure PPy film B"),
    ("4dunk_pol_A_LiClO4", "4-dunk (A)"),
    ("4dunk_pol_B_LiClO4", "4-dunk (B)"),
    ("2chem_pol_old_method_first_test_LiClO4_PC", "2x chem OLD METHOD"),
    ("electropol_on_2chem_pol_LiClO4", "electropol on 2x chem"),
    ("3chem_pol_A_LiClO4", "3x chem pol (A)"),
    ("2chem_pol_B_LiClO4", "2x chem pol (B)"),
    ("decell_A_LiClO4", "decell only (0% PPy)"),
    ("platinum_baseline_LiClO4", "Pt blank"),
]
results = [dict(fit(n), label=lab) for n, lab in SAMPLES]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, r in zip(axes.ravel(), results):
    ax.plot(r["Z"].real, -r["Z"].imag, "o", ms=4, color="#4c72b0", label="data", zorder=3)
    wf = 2 * np.pi * np.logspace(np.log10(r["f"].min()), np.log10(r["f"].max()), 600)
    Zf = model(r["p"], wf)
    ax.plot(Zf.real, -Zf.imag, "-", color="#c44e52", lw=1.9, label="fit (+stray C)", zorder=4)
    ax.set_title(f"{r['label']}  ({r['ts']})", fontsize=9)
    ax.set_xlabel("Re(Z) / Ohm", fontsize=8)
    ax.set_ylabel("-Im(Z) / Ohm", fontsize=8)
    rct = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    ax.text(0.04, 0.96, f"Rs={r['Rs']:.0f}\nRct={rct}\nndl={r['ndl']:.2f}\np_w={r['pw']:.2f}\nCp={r['Cp']*1e9:.1f}nF\nfit {r['rmse']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.9))
    ax.legend(fontsize=6.2, loc="lower right")
    ax.grid(alpha=.3)
fig.suptitle("Randles + parasitic parallel C  -  LiClO4/PC (full spectrum, no trimming)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT}/10_randles_strayC.png", dpi=110)
plt.close()

print(f"{'sample':<24}{'time':>9}{'Rs':>7}{'Rct':>9}{'ndl':>6}{'p_w':>6}{'Cp/nF':>7}{'fit%':>7}")
for r in results:
    rct = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    print(f"{r['label']:<24}{r['ts']:>9}{r['Rs']:>7.0f}{rct:>9}{r['ndl']:>6.2f}{r['pw']:>6.2f}{r['Cp']*1e9:>7.1f}{r['rmse']:>7.1f}")
