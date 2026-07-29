"""Polymerisation-method ladder: EIS vs PPy loading, 2026-07-23 session.

Adds the Pt blank, the decell-only control, the 4-dunk method, 3x chem pol and
the electropolymerised-on-chem-pol sample to the earlier comparison.

PPy mass fractions come from the weight sheets in ../7.15.26_thin_ppy_chicken_experiments/.
Run from this directory:  python3 eis_ladder.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from galvani import BioLogic as BL

OUT = "plots"
WEIGHTS = "../7.15.26_thin_ppy_chicken_experiments"
F_EVAL = 0.05                      # 50 mHz: the lowest frequency common to every run
os.makedirs(OUT, exist_ok=True)

load = lambda n: np.array(BL.MPRfile(n + "_C01.mpr").data)
rs_of = lambda d: d["Re(Z)/Ohm"][np.argmax(d["freq/Hz"])]


def at(d, col, f0):
    f = d["freq/Hz"][::-1]
    return np.interp(np.log10(f0), np.log10(f), d[col][::-1])


# name, label, colour.  Ordered blank -> control -> constructs -> pure film.
LICLO4 = [("platinum_baseline_LiClO4", "Pt blank", "#7f7f7f"),
          ("decell_A_LiClO4", "decell only (0% PPy)", "#8c564b"),
          ("2chem_pol_B_LiClO4", "2x chem pol (B)", "#d62728"),
          ("3chem_pol_A_LiClO4", "3x chem pol (A)", "#ff7f0e"),
          ("2chem_pol_old_method_first_test_LiClO4_PC", "2x chem OLD METHOD", "#2ca02c"),
          ("electropol_on_2chem_pol_LiClO4", "electropol on 2x chem", "#9467bd"),
          ("4dunk_pol_A_LiClO4", "4-dunk (A)", "#1f77b4"),
          ("4dunk_pol_B_LiClO4", "4-dunk (B)", "#17becf"),
          ("pure_ppy_first_test_0", "pure PPy film A", "#000000")]
FECN = [("platinum_baseline_FeCN", "Pt blank", "#7f7f7f"),
        ("decell_A_LCN", "decell only (0% PPy)", "#8c564b"),
        ("2chem_pol_B_FeCN", "2x chem pol (B)", "#d62728"),
        ("2chem_pol_old_method_second_test_FeCN", "2x chem OLD METHOD", "#2ca02c"),
        ("electropol_on_2chem_pol_FeCN", "electropol on 2x chem", "#9467bd"),
        ("4dunk_pol_A_FeCN", "4-dunk (A)", "#1f77b4"),
        ("4dunk_pol_B_FeCN", "4-dunk (B)", "#17becf"),
        ("pure_ppy_fourth_test_FeCN", "pure PPy film A", "#000000")]

# ------------------------------------------------------------------ spectra
fig, ax = plt.subplots(2, 3, figsize=(17.5, 10))
for r, (grp, ttl) in enumerate([(LICLO4, "LiClO4/PC (0 V vs Eoc)"),
                                (FECN, "aq. FeCN (+0.23 V vs Ref)")]):
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
        a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/5_ladder_spectra.png", dpi=115)
plt.close()

# ------------------------------------------------------------------ summary
def table(grp):
    return {lab: rs_of(load(fn)) for fn, lab, _ in grp}

rs_lic, rs_fec = table(LICLO4), table(FECN)
shared = [l for l in rs_lic if l in rs_fec]

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5))

# bar chart, both electrolytes, decell control marked
x = np.arange(len(shared))
ax[0].bar(x - .2, [rs_lic[l] for l in shared], .4, label="LiClO4/PC", color="#4c72b0")
ax[0].bar(x + .2, [rs_fec[l] for l in shared], .4, label="aq. FeCN", color="#dd8452")
ax[0].axhline(rs_lic["decell only (0% PPy)"], ls="--", lw=1.2, color="#4c72b0", alpha=.8)
ax[0].axhline(rs_fec["decell only (0% PPy)"], ls="--", lw=1.2, color="#dd8452", alpha=.8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(shared, rotation=35, ha="right", fontsize=7.5)
ax[0].set(ylabel="Rs / Ohm", yscale="log", title="Series resistance (dashed = decell control)")
ax[0].legend(fontsize=8)

# do the two electrolytes agree on the ranking?
xs = [rs_lic[l] for l in shared]
ys = [rs_fec[l] for l in shared]
ax[1].loglog(xs, ys, "o", ms=8, color="#c44e52")
for l, a_, b_ in zip(shared, xs, ys):
    ax[1].annotate(l, (a_, b_), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
rho = np.corrcoef(np.log10(xs), np.log10(ys))[0, 1]
ax[1].set(xlabel="Rs in LiClO4/PC / Ohm", ylabel="Rs in aq. FeCN / Ohm",
          title=f"Cross-electrolyte consistency (log-log r = {rho:.2f})")
ax[1].grid(alpha=.3, which="both")

# PPy loading vs conductivity
d4 = pd.read_csv(f"{WEIGHTS}/07.23-.26_pyyrole chickenbots weights - dunking_chem_pol_method.csv")
d2 = pd.read_csv(f"{WEIGHTS}/07.20.26-pyyrole chickenbots weights - Sheet1.csv")
m = d4["Decellularized (wet weight)"].notna() & d4["Decellularized (dry weight)"].notna()
FRAC = d4.loc[m, "Decellularized (dry weight)"].sum() / d4.loc[m, "Decellularized (wet weight)"].sum()


def ppy_pct(wet, dry):
    k = wet.notna() & dry.notna()
    pct = 100 * (dry[k] - wet[k] * FRAC) / dry[k]
    return pct[(dry[k] <= wet[k]) & (pct > -50) & (pct < 100)]


load_map = {"4-dunk (A)": ppy_pct(d4["Decellularized (wet weight)"], d4["4 Dunks - Chem Pol (dry weight)"]),
            "2x chem pol (B)": ppy_pct(d2["Decellularized (wet weight)"], d2["Chem Pol Round 2 (dry weight)"])}
for lab, pct in load_map.items():
    ax[2].errorbar(np.median(pct), rs_lic[lab], xerr=[[np.median(pct) - pct.min()], [pct.max() - np.median(pct)]],
                   fmt="o", ms=9, capsize=4, label=f"{lab}  (n={len(pct)})")
ax[2].scatter([0], [rs_lic["decell only (0% PPy)"]], s=90, marker="s", color="#8c564b", label="decell control (0%)")
ax[2].set(xlabel="PPy mass fraction of dry construct / %", ylabel="Rs in LiClO4/PC / Ohm", yscale="log",
          title="More PPy does not mean more conductive")
ax[2].grid(alpha=.3)
ax[2].legend(fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUT}/6_ladder_summary.png", dpi=115)
plt.close()

# ------------------------------------------------------------------ printout
print(f"decell dry fraction used: {FRAC:.4f}\n")
print(f"{'sample':<26}{'Rs LiClO4':>11}{'Rs FeCN':>10}{'vs decell (PC)':>16}")
for l in shared:
    print(f"{l:<26}{rs_lic[l]:>11.0f}{rs_fec[l]:>10.0f}{rs_lic[l]/rs_lic['decell only (0% PPy)']:>15.2f}x")
print(f"\nmedian PPy mass fraction: 4-dunk {np.median(load_map['4-dunk (A)']):.0f}%  "
      f"2x chem pol {np.median(load_map['2x chem pol (B)']):.0f}%")
print(f"cross-electrolyte log-log correlation r = {rho:.3f}")
