"""EIS / CV analysis for the 2026-07-23 impedance session (BioLogic SP-300).

Reads the raw .mpr files with `galvani` (pip install galvani) and writes the
overview figures into plots/.  Run from this directory:  python3 eis_analysis.py

Sample key
----------
  pure_ppy_{first,second}_test        pure PPy film A, LiClO4/PC
  pure_ppy_{third,fourth}_test_FeCN   pure PPy film A, ferri/ferrocyanide
  pure_ppy_B_*                        pure PPy film B
  2chem_pol_old_method_*              chem-polymerised decell construct, OLD method
  2chem_pol_B_*                       chem-polymerised decell construct, sample B

NOTE: EC-Lab electrode area is left at the 0.001 cm2 placeholder in every .mps,
so nothing here is area-normalised.  All comparisons are per-sample-as-mounted.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from galvani import BioLogic as BL

OUT = "plots"
os.makedirs(OUT, exist_ok=True)

load = lambda f: np.array(BL.MPRfile(f).data)


def rs_of(d):
    """Series resistance = Re(Z) at the highest measured frequency."""
    return d["Re(Z)/Ohm"][np.argmax(d["freq/Hz"])]


def interp_at(d, col, f0):
    f = d["freq/Hz"][::-1]
    return np.interp(np.log10(f0), np.log10(f), d[col][::-1])


# ---------------------------------------------------------------- figure 1
# chem-pol B: EIS before CV -> CV -> EIS after CV
pre = load("2chem_pol_B_LiClO4_C01.mpr")
post = load("2chem_pol_B_LiClO4_v2_C01.mpr")
cv = load("2chem_pol_B_LiClO4_CV_C01.mpr")

fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
for d, lab, c in [(pre, "EIS 1 - as-received (14:30)", "#1f77b4"),
                  (post, "EIS 2 - after 2x CV (14:46)", "#d62728")]:
    ax[0, 0].plot(d["Re(Z)/Ohm"], d["-Im(Z)/Ohm"], "o-", ms=3.5, lw=.9, color=c, label=lab)
    ax[0, 1].loglog(d["freq/Hz"], d["|Z|/Ohm"], "o-", ms=3.5, lw=.9, color=c, label=lab)
    ax[0, 2].semilogx(d["freq/Hz"], d["Phase(Z)/deg"], "o-", ms=3.5, lw=.9, color=c, label=lab)
ax[0, 0].set(xlabel="Re(Z) / Ohm", ylabel="-Im(Z) / Ohm", title="Nyquist - chem-pol B, before vs after CV")
ax[0, 0].axis("equal")
ax[0, 1].set(xlabel="f / Hz", ylabel="|Z| / Ohm", title="Bode magnitude")
ax[0, 2].set(xlabel="f / Hz", ylabel="Phase / deg", title="Bode phase")

cyc = cv["cycle number"]
for k in np.unique(cyc):
    m = cyc == k
    ax[1, 0].plot(cv["Ewe/V"][m], cv["<I>/mA"][m], lw=1.1, label=f"cycle {int(k)}")
ax[1, 0].set(xlabel="Ewe / V vs Ref", ylabel="I / mA", title="CV - chem-pol B, LiClO4/PC, 100 mV/s")
ax[1, 0].axhline(0, color="k", lw=.5)

ax[1, 1].plot(cv["time/s"], cv["Ewe/V"], lw=.9, color="C0")
ax[1, 1].set(xlabel="t / s", ylabel="Ewe / V", title="CV potential program")
tw = ax[1, 1].twinx()
tw.plot(cv["time/s"], cv["<I>/mA"], lw=.7, color="C3", alpha=.7)
tw.set_ylabel("I / mA", color="C3")

g = np.logspace(np.log10(max(pre["freq/Hz"].min(), post["freq/Hz"].min())),
                np.log10(min(pre["freq/Hz"].max(), post["freq/Hz"].max())), 200)
ratio = interp_at(post, "|Z|/Ohm", g) / interp_at(pre, "|Z|/Ohm", g)
ax[1, 2].semilogx(g, ratio, lw=1.6, color="k")
ax[1, 2].axhline(1, ls="--", color="gray")
ax[1, 2].set(xlabel="f / Hz", ylabel="|Z| after / |Z| before", title="Change induced by CV priming")

for a in ax.ravel():
    a.grid(alpha=.3)
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/1_chempolB_before_after_CV.png", dpi=115)
plt.close()

# ---------------------------------------------------------------- figure 2
# cross-sample, split by electrolyte
liclo4 = [("pure_ppy_first_test_0_C01.mpr", "pure PPy A (run1)", "#1f77b4"),
          ("pure_ppy_second_test_C01.mpr", "pure PPy A (run2)", "#aec7e8"),
          ("pure_ppy_B_LiClO4_PC_v2_C01.mpr", "pure PPy B", "#17becf"),
          ("2chem_pol_old_method_first_test_LiClO4_PC_C01.mpr", "chem-pol OLD METHOD", "#2ca02c"),
          ("2chem_pol_B_LiClO4_C01.mpr", "chem-pol B (pre-CV)", "#d62728"),
          ("2chem_pol_B_LiClO4_v2_C01.mpr", "chem-pol B (post-CV)", "#ff9896")]
fecn = [("pure_ppy_third_test_FeCN_C01.mpr", "pure PPy A (run3)", "#1f77b4"),
        ("pure_ppy_fourth_test_FeCN_C01.mpr", "pure PPy A (run4)", "#aec7e8"),
        ("pure_ppy_B_FeCN_C01.mpr", "pure PPy B", "#17becf"),
        ("2chem_pol_old_method_second_test_FeCN_C01.mpr", "chem-pol OLD METHOD (r2)", "#2ca02c"),
        ("2chem_pol_old_method_third_test_FeCN_C01.mpr", "chem-pol OLD METHOD (r3)", "#98df8a")]

fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
for r, (grp, ttl) in enumerate([(liclo4, "LiClO4/PC (0 V vs Eoc)"),
                                (fecn, "FeCN (+0.23 V vs Ref)")]):
    for fn, lab, c in grp:
        d = load(fn)
        im = d["-Im(Z)/Ohm"]
        ax[r, 0].loglog(d["Re(Z)/Ohm"], np.where(im > 0, im, np.nan), "o-", ms=3, lw=.9, color=c, label=lab)
        ax[r, 1].loglog(d["freq/Hz"], d["|Z|/Ohm"], "o-", ms=3, lw=.9, color=c, label=lab)
        ax[r, 2].semilogx(d["freq/Hz"], d["Phase(Z)/deg"], "o-", ms=3, lw=.9, color=c, label=lab)
    ax[r, 0].set(xlabel="Re(Z) / Ohm", ylabel="-Im(Z) / Ohm", title=f"Nyquist (log-log) - {ttl}")
    ax[r, 1].set(xlabel="f / Hz", ylabel="|Z| / Ohm", title=f"Bode magnitude - {ttl}")
    ax[r, 2].set(xlabel="f / Hz", ylabel="Phase / deg", title=f"Bode phase - {ttl}")
    for a in ax[r]:
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=7.5)
plt.tight_layout()
plt.savefig(f"{OUT}/2_cross_sample.png", dpi=115)
plt.close()

# ---------------------------------------------------------------- figure 3
# Rs-subtracted Nyquist: the interfacial response, with the series R removed
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for fn, lab, c in liclo4:
    d = load(fn)
    re_ = d["Re(Z)/Ohm"] - rs_of(d)
    im = d["-Im(Z)/Ohm"]
    ax[0].plot(re_, im, "o-", ms=3.5, lw=.9, color=c, label=lab)
    ax[1].plot(re_, im, "o-", ms=3.5, lw=.9, color=c, label=lab)
    ax[2].loglog(d["freq/Hz"], np.where(im > 0, im, np.nan), "o-", ms=3, lw=.9, color=c, label=lab)
ax[0].set(xlabel="Re(Z) - Rs / Ohm", ylabel="-Im(Z) / Ohm", title="Rs-subtracted Nyquist (full)")
ax[1].set(xlim=(-20, 900), ylim=(-20, 400), xlabel="Re(Z) - Rs / Ohm", ylabel="-Im(Z) / Ohm",
          title="Rs-subtracted Nyquist (zoom on arcs)")
ax[2].set(xlabel="f / Hz", ylabel="-Im(Z) / Ohm", title="Imaginary part vs frequency")
for a in ax:
    a.grid(alpha=.3)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/3_rs_subtracted.png", dpi=115)
plt.close()

# ---------------------------------------------------------------- figure 4
# the chem-pol B CV is IR-drop limited; correct it with Ru from EIS
Ru = rs_of(pre)
Ecorr = cv["Ewe/V"] - (cv["<I>/mA"] / 1000.0) * Ru
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for k in np.unique(cyc):
    m = cyc == k
    ax[0].plot(cv["Ewe/V"][m], cv["<I>/mA"][m], lw=1.1, label=f"cycle {int(k)}")
    ax[1].plot(Ecorr[m], cv["<I>/mA"][m], lw=1.1, label=f"cycle {int(k)}")
ax[0].set(xlabel="Ewe applied / V vs Ref", ylabel="I / mA", title="CV as recorded")
ax[1].set(xlabel="Ewe - I*Ru / V vs Ref", ylabel="I / mA",
          title=f"CV IR-corrected (Ru={Ru:.0f} Ohm from EIS)")
for a in ax:
    a.grid(alpha=.3)
    a.axhline(0, color="k", lw=.5)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/4_cv_IR_corrected.png", dpi=115)
plt.close()

# ---------------------------------------------------------------- tables
SOLUTION_FLOOR = 245.0  # Rs of pure PPy A in LiClO4/PC: best estimate of the cell/solution floor

print("== LiClO4/PC, matched frequencies ==")
print(f"{'sample':<28}{'Rs':>7}{'|Z|@1Hz':>9}{'|Z|@0.1Hz':>10}{'ph@0.1Hz':>9}{'Z0.1/Rs':>9}{'Rs-floor':>9}")
for fn, lab, _ in liclo4:
    d = load(fn)
    rs = rs_of(d)
    z1 = interp_at(d, "|Z|/Ohm", 1.0)
    z01 = interp_at(d, "|Z|/Ohm", 0.1)
    p01 = interp_at(d, "Phase(Z)/deg", 0.1)
    print(f"{lab:<28}{rs:>7.0f}{z1:>9.0f}{z01:>10.0f}{p01:>9.1f}{z01/rs:>9.2f}{rs-SOLUTION_FLOOR:>9.0f}")

print("\n== FeCN, matched frequencies ==")
print(f"{'sample':<28}{'Rs':>7}{'|Z|@1Hz':>9}{'|Z|@0.2Hz':>10}{'ph@0.2Hz':>9}{'Z0.2/Rs':>9}")
for fn, lab, _ in fecn:
    d = load(fn)
    rs = rs_of(d)
    z1 = interp_at(d, "|Z|/Ohm", 1.0)
    z02 = interp_at(d, "|Z|/Ohm", 0.2)
    p02 = interp_at(d, "Phase(Z)/deg", 0.2)
    print(f"{lab:<28}{rs:>7.0f}{z1:>9.0f}{z02:>10.0f}{p02:>9.1f}{z02/rs:>9.2f}")

print("\n== chem-pol B CV: ohmic fit per cycle ==")
for k in np.unique(cyc):
    m = cyc == k
    p = np.polyfit(cv["Ewe/V"][m], cv["<I>/mA"][m], 1)
    pred = np.polyval(p, cv["Ewe/V"][m])
    r2 = 1 - np.sum((cv["<I>/mA"][m] - pred) ** 2) / np.sum((cv["<I>/mA"][m] - cv["<I>/mA"][m].mean()) ** 2)
    print(f"  cycle {int(k)}: R={1000/p[0]:7.0f} Ohm   E(I=0)={-p[1]/p[0]:+.3f} V   R^2={r2:.5f}")
print(f"  max IR drop = {np.abs(cv['<I>/mA']/1000*Ru).max():.3f} V of the 1.400 V applied window")
print(f"  true window at the film: {Ecorr.min():+.3f} .. {Ecorr.max():+.3f} V")
