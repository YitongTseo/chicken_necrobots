import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FN = "07-28-26_CV_on_New_Electropol_Cell_AND_sample.csv"

# File is UTF-16. Header rows precede the data. Column order is scan 1,10,2,3,...,9
# each scan = (V, uA) pair.
raw = pd.read_csv(FN, encoding="utf-16", skiprows=4, header=None)

# Drop fully empty columns (the export puts blank separator columns)
raw = raw.dropna(axis=1, how="all")

# The scan order in the header
scan_order = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

scans = {}
ncols = raw.shape[1]
for i, s in enumerate(scan_order):
    vcol = 2 * i
    icol = 2 * i + 1
    if icol >= ncols:
        break
    V = pd.to_numeric(raw.iloc[:, vcol], errors="coerce").values
    I = pd.to_numeric(raw.iloc[:, icol], errors="coerce").values
    mask = ~np.isnan(V) & ~np.isnan(I)
    scans[s] = (V[mask], I[mask])

print("Scans parsed:", sorted(scans.keys()))
for s in sorted(scans.keys()):
    V, I = scans[s]
    print(f"Scan {s:2d}: n={len(V):4d}  V[{V.min():.3f},{V.max():.3f}]  "
          f"I[{I.min():.1f},{I.max():.1f}] uA")

# Scan rate & window sanity
V1, I1 = scans[1]
print("\nV start:", V1[0], "V end:", V1[-1])
print("Vertex (min V):", V1.min(), " max V:", V1.max())

# ---- Plot all cycles ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
cmap = plt.cm.viridis
order_plot = [1,2,3,4,5,6,7,8,9,10]
for s in order_plot:
    V, I = scans[s]
    c = cmap((s-1)/9)
    ax[0].plot(V, I, color=c, lw=1, label=f"cycle {s}")
ax[0].set_xlabel("Potential E (V)")
ax[0].set_ylabel("Current (uA)")
ax[0].set_title("All 10 CV cycles (25 mV/s)")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].legend(fontsize=8, ncol=2)

# First vs last
for s, lab, col in [(1,"cycle 1 (first)","tab:red"),(10,"cycle 10 (last)","tab:blue")]:
    V, I = scans[s]
    ax[1].plot(V, I, col, lw=1.3, label=lab)
ax[1].set_xlabel("Potential E (V)")
ax[1].set_ylabel("Current (uA)")
ax[1].set_title("First vs last cycle")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].legend()
plt.tight_layout()
plt.savefig("cv_overview.png", dpi=130)
print("\nsaved cv_overview.png")

# ---- Separate anodic (forward, increasing... actually sweep starts high) ----
# Determine sweep direction by dV
def split_branches(V, I):
    dV = np.gradient(V)
    fwd = dV < 0   # going from +0.9 down to -0.6 (cathodic-going)
    rev = dV > 0   # going back up (anodic-going)
    return fwd, rev

# Peak / current summary per cycle
print("\nPer-cycle summary:")
print(f"{'cyc':>3} {'Imax(uA)':>9} {'E@Imax':>7} {'Imin(uA)':>9} {'E@Imin':>7} "
      f"{'I@0.9V':>8} {'I@-0.6V':>8}")
rows=[]
for s in order_plot:
    V, I = scans[s]
    imax_idx = np.argmax(I); imin_idx = np.argmin(I)
    # current near anodic vertex (+0.9) and cathodic vertex (-0.6)
    i_at_high = I[np.argmin(np.abs(V - V.max()))]
    i_at_low  = I[np.argmin(np.abs(V - V.min()))]
    print(f"{s:>3} {I[imax_idx]:9.1f} {V[imax_idx]:7.3f} {I[imin_idx]:9.1f} "
          f"{V[imin_idx]:7.3f} {i_at_high:8.1f} {i_at_low:8.1f}")
    rows.append((s, I[imax_idx], V[imax_idx], I[imin_idx], V[imin_idx]))
