"""(1) draw the fitted equivalent circuit, (2) fit the v2 model to BOTH electrolytes,
(3) plot fitted Rs per sample type (individual replicate points, no averaging).

Circuit:  Cp  ||  [ Rs --- ( CPE_dl || (Rct --- Warburg) ) ]
  Cp    stray/parasitic capacitance across the cell (cable+cell+reference)
  Rs    series resistance (electrolyte + clamp/contact)
  CPE   double-layer (constant-phase element)
  Rct   charge-transfer resistance
  W     Warburg (diffusion)

FeCN spectra are masked to f >= 1 Hz (sub-1 Hz points are corrupted by clamp-iron
contamination -> negative -Im, scattered phase).  Run:  python3 eis_final.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy.optimize import least_squares
from galvani import BioLogic as BL

OUT = "plots"
os.makedirs(OUT, exist_ok=True)
load = lambda n: np.array(BL.MPRfile(n + "_C01.mpr").data)


def model(p, w):
    Rs, Qdl, ndl, Rct, Aw, pw, Cp = p
    jw = 1j * w
    Zw = Aw * jw ** (-pw)
    Yr = Qdl * jw ** ndl + 1.0 / (Rct + Zw)
    Zr = Rs + 1.0 / Yr
    return 1.0 / (1j * w * Cp + 1.0 / Zr)


def fit(name, fmin=None):
    d = load(name)
    f = d["freq/Hz"]
    keep = np.isfinite(f)
    if fmin is not None:
        keep &= f >= fmin
    f = f[keep]
    Z = d["Re(Z)/Ohm"][keep] - 1j * d["-Im(Z)/Ohm"][keep]
    w = 2 * np.pi * f
    Rs0 = d["Re(Z)/Ohm"][np.argmax(d["freq/Hz"])]
    span = max(d["Re(Z)/Ohm"].max() - Rs0, 20.0)
    Rct0 = span * 0.6
    fap = f[np.argmax(d["-Im(Z)/Ohm"][keep])]
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
    return dict(Rs=p[0], Rct=p[3], Cp=p[6], rmse=rmse)


# (type, replicate, LiClO4 name, FeCN name)
SAMPLES = [
    ("pure PPy", "A", "pure_ppy_first_test_0", "pure_ppy_fourth_test_FeCN"),
    ("pure PPy", "B", "pure_ppy_B_LiClO4_PC_v2", "pure_ppy_B_FeCN"),
    ("4-dunk", "A", "4dunk_pol_A_LiClO4", "4dunk_pol_A_FeCN"),
    ("4-dunk", "B", "4dunk_pol_B_LiClO4", "4dunk_pol_B_FeCN"),
    ("2x chem pol", "A", "2chem_pol_old_method_first_test_LiClO4_PC", "2chem_pol_old_method_second_test_FeCN"),
    ("2x chem pol", "B", "2chem_pol_B_LiClO4", "2chem_pol_B_FeCN"),
    ("3x chem pol", "A", "3chem_pol_A_LiClO4", "3chem_pol_A_FeCN"),
    ("electropol\non 2x chem", "A", "electropol_on_2chem_pol_LiClO4", "electropol_on_2chem_pol_FeCN"),
    ("decell\n(0% PPy)", "A", "decell_A_LiClO4", "decell_A_LCN"),
    ("Pt blank", "-", "platinum_baseline_LiClO4", "platinum_baseline_FeCN"),
]

records = []
for typ, rep, lic, fec in SAMPLES:
    rl = fit(lic, fmin=None)
    rf = fit(fec, fmin=1.0)
    records.append(dict(typ=typ, rep=rep, lic=rl, fec=rf))

# =========================================================== FIG 11: circuit
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
LW = 2.0


def wire(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color="k", lw=LW, zorder=1, solid_capstyle="round")


def box(xc, yc, label, sub="", w=1.3, h=0.6, fc="#eaf2fb"):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                 boxstyle="round,pad=0.02", fc=fc, ec="k", lw=LW, zorder=3))
    ax.text(xc, yc + 0.06, label, ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    if sub:
        ax.text(xc, yc - 0.17, sub, ha="center", va="center", fontsize=8.5, zorder=4, color="#333")


def cap(xc, yc, label, sub="", plate=0.34, gap=0.22):
    for dx in (-gap / 2, gap / 2):
        ax.plot([xc + dx, xc + dx], [yc - plate, yc + plate], color="k", lw=2.4, zorder=3)
    ax.text(xc, yc + plate + 0.28, label, ha="center", fontsize=13, fontweight="bold")
    ax.text(xc, yc + plate + 0.02, sub, ha="center", fontsize=8.5, color="#333")


yM, yT, yB, yC = 3.0, 3.9, 2.1, 5.1
xA, xB = 1.0, 9.8
# main series rail: A - Rs - node1 ... node2 - B
wire(xA, yM, 2.0, yM)
box(2.6, yM, "R$_s$", "series R\n(soln+clamp)")
wire(3.25, yM, 4.3, yM)              # to node1
n1, n2 = 4.3, 8.3
# parallel block: node1 up to CPE, down to Rct+W, rejoin at node2
wire(n1, yM, n1, yT); wire(n1, yT, 5.4, yT)
box(6.1, yT, "CPE$_{dl}$", "double layer")
wire(6.75, yT, n2, yT); wire(n2, yT, n2, yM)
wire(n1, yM, n1, yB); wire(n1, yB, 5.05, yB)
box(5.6, yB, "R$_{ct}$", "charge transfer", w=1.0)
wire(6.1, yB, 6.5, yB)
box(7.1, yB, "W", "Warburg\n(diffusion)", w=1.0)
wire(7.6, yB, n2, yB); wire(n2, yB, n2, yM)
wire(n1, yM, n1, yM)                 # node dot
wire(n2, yM, xB, yM)                 # node2 to B
# stray Cp across A-B (top rail)
wire(xA, yM, xA, yC); wire(xA, yC, 5.0, yC)
cap(5.4, yC, "C$_p$", "stray / parasitic")
wire(5.8, yC, xB, yC); wire(xB, yC, xB, yM)
# node dots
for (xx, yy) in [(n1, yM), (n2, yM), (xA, yM), (xB, yM)]:
    ax.plot(xx, yy, "o", color="k", ms=6, zorder=5)
# terminals
ax.plot(xA, yM, "o", mfc="white", mec="k", ms=13, mew=2, zorder=6)
ax.plot(xB, yM, "o", mfc="white", mec="k", ms=13, mew=2, zorder=6)
ax.text(xA, yM - 0.55, "working\nelectrode", ha="center", fontsize=9.5, style="italic")
ax.text(xB, yM - 0.55, "counter /\nreference", ha="center", fontsize=9.5, style="italic")
ax.text(5.5, 0.7, r"$Z(\omega)=\dfrac{1}{\,j\omega C_p + 1/Z_r\,}$,   "
        r"$Z_r=R_s+\dfrac{1}{\,Q(j\omega)^{n}+1/(R_{ct}+A_W(j\omega)^{-p})\,}$",
        ha="center", fontsize=13,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#ccc"))
ax.set_title("Fitted equivalent circuit:  stray $C_p$ in parallel with a Randles cell",
             fontsize=14, pad=8)
plt.tight_layout()
plt.savefig(f"{OUT}/11_circuit.png", dpi=130)
plt.close()

# =========================================================== FIG 12: Rs by type
plot_records = list(records)
# order sample types left->right by mean fitted Rs (LiClO4) = low resistance first
type_rs = {}
for r in plot_records:
    type_rs.setdefault(r["typ"], []).append(r["lic"]["Rs"])
types = sorted(type_rs, key=lambda t: np.mean(type_rs[t]))
xpos = {t: i for i, t in enumerate(types)}

fig, ax = plt.subplots(figsize=(13, 6.5))
for elec, key, col, dx in [("LiClO4/PC", "lic", "#2c6fbb", -0.13), ("aq. FeCN", "fec", "#e07b39", +0.13)]:
    # group replicates within a type to spread them horizontally
    per_type = {}
    for r in plot_records:
        per_type.setdefault(r["typ"], []).append(r)
    first = True
    for r in plot_records:
        sibs = per_type[r["typ"]]
        x = xpos[r["typ"]] + dx
        y = r[key]["Rs"]
        ax.scatter(x, y, s=130, color=col, edgecolor="k", lw=.8, zorder=3,
                   label=elec if first else None)
        first = False
        if r["rep"] not in ("-", "A") or len(sibs) > 1:
            ax.annotate(r["rep"], (x, y), fontsize=7.5, xytext=(0, 9),
                        textcoords="offset points", ha="center", color=col)
ax.set_xticks(range(len(types)))
ax.set_xticklabels(types, fontsize=9.5)
ax.set_yscale("log")
ax.set_ylabel("fitted series resistance  R$_s$  / Ohm", fontsize=11)
ax.set_title("Fitted R$_s$ by sample type  (individual runs)", fontsize=13)
ax.grid(axis="y", alpha=.3, which="both")
ax.legend(fontsize=10, title="electrolyte")
plt.tight_layout()
plt.savefig(f"{OUT}/12_Rs_by_type.png", dpi=120)
plt.close()

# =========================================================== printout
print(f"{'type':<16}{'rep':<12}{'Rs LiClO4':>11}{'Rs FeCN':>10}{'fitLiC%':>9}{'fitFeCN%':>10}")
for r in records:
    print(f"{r['typ'].replace(chr(10),' '):<16}{r['rep']:<12}{r['lic']['Rs']:>11.0f}{r['fec']['Rs']:>10.0f}"
          f"{r['lic']['rmse']:>9.1f}{r['fec']['rmse']:>10.1f}")
