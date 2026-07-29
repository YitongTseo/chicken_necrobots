"""Fit a simple Randles-type model R-(RQ) to every EIS run, and test whether
the corroding clamp introduced a time-dependent bias across the session.

Model (per spectrum):   Z(w) = Rs + Rct / (1 + Rct*Q*(j w)^n)
  Rs   series (solution + clamp/contact) resistance      [Ohm]
  Rct  charge-transfer / interfacial resistance          [Ohm]
  Q,n  constant-phase element (n=1 -> ideal capacitor)
  C_eff = (Q * Rct)^(1/n) / Rct   (Brug pseudo-capacitance)

Non-physical points (-Im(Z) <= 0) are dropped before fitting; proportional
weighting (1/|Z|) is used, standard for EIS.  Run:  python3 eis_fit.py
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


def timestamp(n):
    return str(getattr(BL.MPRfile(n + "_C01.mpr"), "timestamp", ""))[11:19]


def model(p, w):
    Rs, Rct, Q, n = p
    return Rs + Rct / (1.0 + Rct * Q * (1j * w) ** n)


def fit(d, fmin=None):
    f = d["freq/Hz"]
    Z = d["Re(Z)/Ohm"] - 1j * d["-Im(Z)/Ohm"]     # note: file stores -Im, so Zimag = -that
    good = d["-Im(Z)/Ohm"] > 0
    if fmin is not None:
        good &= f >= fmin
    f, Z = f[good], Z[good]
    w = 2 * np.pi * f
    Rs0 = d["Re(Z)/Ohm"][np.argmax(d["freq/Hz"])]
    Rct0 = max(d["Re(Z)/Ohm"].max() - Rs0, 10.0)
    fmid = f[len(f) // 2]
    Q0 = 1.0 / (Rct0 * (2 * np.pi * fmid))
    p0 = [Rs0, Rct0, Q0, 0.85]

    def resid(lp):
        p = np.exp(lp[:3]).tolist() + [lp[3]]
        Zm = model(p, w)
        r = (Zm - Z) / np.abs(Z)
        return np.concatenate([r.real, r.imag])

    lp0 = [np.log(p0[0]), np.log(p0[1]), np.log(p0[2]), p0[3]]
    res = least_squares(resid, lp0, bounds=([np.log(1), np.log(1), np.log(1e-12), 0.2],
                                            [np.log(1e6), np.log(1e9), np.log(1e2), 1.0]),
                        method="trf", max_nfev=20000)
    p = np.exp(res.x[:3]).tolist() + [res.x[3]]
    Rs, Rct, Q, n = p
    Ceff = (Q * Rct) ** (1.0 / n) / Rct
    rmse = np.sqrt(np.mean(res.fun ** 2)) * 100
    return dict(Rs=Rs, Rct=Rct, Q=Q, n=n, Ceff=Ceff, rmse=rmse, f=f, Z=Z, w=w, p=p)


# name, label, electrolyte, fmin-for-fit (FeCN sub-1Hz is corrupt -> mask)
SAMPLES = [
    ("pure_ppy_first_test_0", "pure PPy film A", "LiClO4", None),
    ("pure_ppy_B_LiClO4_PC_v2", "pure PPy film B", "LiClO4", None),
    ("2chem_pol_B_LiClO4", "2x chem pol (B)", "LiClO4", None),
    ("3chem_pol_A_LiClO4", "3x chem pol (A)", "LiClO4", None),
    ("2chem_pol_old_method_first_test_LiClO4_PC", "2x chem OLD METHOD", "LiClO4", None),
    ("electropol_on_2chem_pol_LiClO4", "electropol on 2x chem", "LiClO4", None),
    ("4dunk_pol_A_LiClO4", "4-dunk (A)", "LiClO4", None),
    ("4dunk_pol_B_LiClO4", "4-dunk (B)", "LiClO4", None),
    ("decell_A_LiClO4", "decell only (0% PPy)", "LiClO4", None),
    ("platinum_baseline_LiClO4", "Pt blank", "LiClO4", None),
    ("pure_ppy_fourth_test_FeCN", "pure PPy film A", "FeCN", 1.0),
    ("pure_ppy_B_FeCN", "pure PPy film B", "FeCN", 1.0),
    ("2chem_pol_B_FeCN", "2x chem pol (B)", "FeCN", 1.0),
    ("3chem_pol_A_FeCN", "3x chem pol (A)", "FeCN", 1.0),
    ("2chem_pol_old_method_second_test_FeCN", "2x chem OLD METHOD", "FeCN", 1.0),
    ("electropol_on_2chem_pol_FeCN", "electropol on 2x chem", "FeCN", 1.0),
    ("4dunk_pol_A_FeCN", "4-dunk (A)", "FeCN", 1.0),
    ("4dunk_pol_B_FeCN", "4-dunk (B)", "FeCN", 1.0),
    ("decell_A_LCN", "decell only (0% PPy)", "FeCN", 1.0),
    ("platinum_baseline_FeCN", "Pt blank", "FeCN", 1.0),
]

results = []
for name, label, elec, fmin in SAMPLES:
    d = load(name)
    r = fit(d, fmin)
    r.update(name=name, label=label, elec=elec, ts=timestamp(name), d=d)
    results.append(r)

# ------------------------------------------------------- FIG 7: Nyquist grid + fits (LiClO4)
lic = [r for r in results if r["elec"] == "LiClO4"]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, r in zip(axes.ravel(), lic):
    d = r["d"]
    ax.plot(d["Re(Z)/Ohm"], d["-Im(Z)/Ohm"], "o", ms=4, color="#4c72b0", label="data", zorder=3)
    wf = 2 * np.pi * np.logspace(np.log10(r["f"].min()), np.log10(r["f"].max()), 400)
    Zf = model(r["p"], wf)
    ax.plot(Zf.real, -Zf.imag, "-", color="#c44e52", lw=1.8, label="R-(RQ) fit")
    ax.set_title(f"{r['label']}  ({r['ts']})", fontsize=9)
    ax.set_xlabel("Re(Z) / Ohm", fontsize=8)
    ax.set_ylabel("-Im(Z) / Ohm", fontsize=8)
    rct_str = ">1e5 (blocking)" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    ax.text(0.04, 0.96, f"Rs={r['Rs']:.0f}\nRct={rct_str}\nn={r['n']:.2f}\nC={r['Ceff']*1e6:.0f}uF\nfit {r['rmse']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.9))
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=.3)
fig.suptitle("R-(RQ) fits — LiClO4/PC (0 V vs Eoc)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT}/7_fits_liclo4.png", dpi=110)
plt.close()

# ------------------------------------------------------- FIG 8: corrosion timeline
def minutes(ts):
    h, m, s = map(int, ts.split(":"))
    return (h - 12) * 60 + m + s / 60.0

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for elec, col in [("LiClO4", "#4c72b0"), ("FeCN", "#dd8452")]:
    pts = [(minutes(r["ts"]), r["Rs"], r["label"]) for r in results if r["elec"] == elec]
    pts.sort()
    xs, ys, labs = zip(*pts)
    ax[0].plot(xs, ys, "o-", color=col, ms=7, lw=.8, label=elec)
    for x, y, l in pts:
        ax[0].annotate(l, (x, y), fontsize=6.3, xytext=(3, 4), textcoords="offset points")
ax[0].set(xlabel="minutes into session (from 12:00)", ylabel="Rs / Ohm", yscale="log",
          title="Series resistance vs wall-clock time")
ax[0].grid(alpha=.3, which="both")
ax[0].legend()

# reference-material floor over time: pure PPy films + Pt (materials that should be low & stable)
refs = [r for r in results if r["label"] in ("pure PPy film A", "pure PPy film B", "Pt blank")]
for elec, col in [("LiClO4", "#4c72b0"), ("FeCN", "#dd8452")]:
    pts = sorted((minutes(r["ts"]), r["Rs"], r["label"]) for r in refs if r["elec"] == elec)
    if pts:
        xs, ys, labs = zip(*pts)
        ax[1].plot(xs, ys, "s-", color=col, ms=9, label=elec)
        for x, y, l in pts:
            ax[1].annotate(l.replace("pure PPy film ", "PPy-").replace("Pt blank", "Pt"),
                           (x, y), fontsize=7, xytext=(3, 5), textcoords="offset points")
ax[1].set(xlabel="minutes into session", ylabel="Rs / Ohm",
          title="Low-resistance reference materials (should be flat if no drift)")
ax[1].grid(alpha=.3)
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/8_corrosion_timeline.png", dpi=115)
plt.close()

# ------------------------------------------------------- printout
print(f"{'sample':<24}{'elec':<7}{'time':>9}{'Rs':>8} | {'Rct':>12} {'n':>6} {'fit%':>6}")
for r in sorted(results, key=lambda r: (r["elec"], minutes(r["ts"]))):
    rct_str = ">1e5" if r["Rct"] > 1e5 else f"{r['Rct']:.0f}"
    print(f"{r['label']:<24}{r['elec']:<7}{r['ts']:>9}{r['Rs']:>8.0f} | {rct_str:>12} {r['n']:>6.2f} {r['rmse']:>6.1f}")

# quantify time-correlation of Rs for the non-reference constructs
from numpy import corrcoef, log10
for elec in ("LiClO4", "FeCN"):
    grp = [r for r in results if r["elec"] == elec and "pure PPy" not in r["label"] and "Pt" not in r["label"]]
    t = [minutes(r["ts"]) for r in grp]
    y = [log10(r["Rs"]) for r in grp]
    print(f"\n{elec}: log10(Rs) vs time correlation over constructs, r = {corrcoef(t, y)[0,1]:+.2f} (n={len(grp)})")
