import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FN = "07-28-26_CV_on_New_Electropol_Cell_AND_sample.csv"
raw = pd.read_csv(FN, encoding="utf-16", skiprows=4, header=None).dropna(axis=1, how="all")
scan_order = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
scans = {}
for i, s in enumerate(scan_order):
    V = pd.to_numeric(raw.iloc[:, 2*i], errors="coerce").values
    I = pd.to_numeric(raw.iloc[:, 2*i+1], errors="coerce").values
    m = ~np.isnan(V) & ~np.isnan(I)
    scans[s] = (V[m], I[m])

def branches(V, I):
    # sweep starts at +0.9, goes down to -0.6 (cathodic-going), then back up (anodic-going)
    dV = np.gradient(V)
    down = dV < 0
    up = dV > 0
    return (V[down], I[down]), (V[up], I[up])

# ---- Figure: anodic branch of each cycle + onset ----
fig, ax = plt.subplots(2, 2, figsize=(15, 11))
cmap = plt.cm.plasma

# (0,0) full CVs
for s in range(1, 11):
    V, I = scans[s]
    ax[0,0].plot(V, I, color=cmap((s-1)/9), lw=1, label=f"cyc {s}")
ax[0,0].axhline(0, color="k", lw=0.5); ax[0,0].axvline(0, color="grey", lw=0.4, ls=":")
ax[0,0].set(xlabel="E (V)", ylabel="I (uA)", title="All CV cycles (25 mV/s, +0.9 to -0.6 V)")
ax[0,0].legend(fontsize=7, ncol=2)

# (0,1) anodic-going branch only (reverse sweep, going up in V)
for s in range(1, 11):
    V, I = scans[s]
    (_,_), (Vu, Iu) = branches(V, I)
    order = np.argsort(Vu)
    ax[0,1].plot(Vu[order], Iu[order], color=cmap((s-1)/9), lw=1)
ax[0,1].axhline(0, color="k", lw=0.5); ax[0,1].axvline(0, color="grey", lw=0.4, ls=":")
ax[0,1].set(xlabel="E (V)", ylabel="I (uA)", title="Anodic-going (upward) branch, all cycles")

# (1,0) cycle 1 with forward/reverse colored and onset marked
V, I = scans[1]
(Vd, Id), (Vu, Iu) = branches(V, I)
ax[1,0].plot(Vd, Id, color="tab:blue", lw=1.2, label="cathodic-going (0.9->-0.6)")
ax[1,0].plot(Vu, Iu, color="tab:red", lw=1.2, label="anodic-going (-0.6->0.9)")
ax[1,0].axhline(0, color="k", lw=0.5)
ax[1,0].set(xlabel="E (V)", ylabel="I (uA)", title="Cycle 1 branches")
ax[1,0].legend(fontsize=8)

# Onset estimate on anodic branch of cycle 1: where dI/dE begins to climb steeply.
# Use the anodic-going branch; find where current rises above baseline capacitive level.
order = np.argsort(Vu)
Vs, Is = Vu[order], Iu[order]
# baseline = median current in the -0.2..+0.2 V window (double-layer region)
base_mask = (Vs > -0.2) & (Vs < 0.2)
baseline = np.median(Is[base_mask])
# onset where current exceeds baseline + threshold and keeps rising
thr = baseline + 10  # uA above baseline
above = Vs[(Is > thr) & (Vs > 0.2)]
onset = above.min() if len(above) else np.nan
ax[1,0].axhline(baseline, color="grey", ls=":", lw=0.8)
if not np.isnan(onset):
    ax[1,0].axvline(onset, color="green", ls="--", lw=1)
    ax[1,0].annotate(f"onset ~{onset:.2f} V", (onset, baseline+15), color="green")

# (1,1) trend of anodic vertex current & cathodic vertex current vs cycle
cyc = np.arange(1,11)
ivtx_hi = [scans[s][1][np.argmin(np.abs(scans[s][0]-scans[s][0].max()))] for s in cyc]
ivtx_lo = [scans[s][1][np.argmin(np.abs(scans[s][0]-scans[s][0].min()))] for s in cyc]
ax[1,1].plot(cyc, ivtx_hi, "o-", color="tab:red", label="I at +0.9 V (anodic vertex)")
ax[1,1].plot(cyc, ivtx_lo, "s-", color="tab:blue", label="I at -0.6 V (cathodic vertex)")
ax[1,1].axhline(0, color="k", lw=0.5)
ax[1,1].set(xlabel="cycle #", ylabel="I (uA)", title="Vertex currents vs cycle")
ax[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("cv_analysis.png", dpi=130)
print("baseline (double-layer) current ~", round(baseline,1), "uA")
print("estimated anodic onset (cycle 1) ~", round(onset,3), "V" if not np.isnan(onset) else "n/a")

# current at candidate hold potentials on cycle-1 anodic branch
print("\nAnodic-going current at candidate hold potentials (cycle 1):")
for Ehold in [0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9]:
    idx = np.argmin(np.abs(Vs-Ehold))
    print(f"  E={Ehold:+.2f} V  ->  I={Is[idx]:7.1f} uA  (above baseline: {Is[idx]-baseline:+.1f})")

# how much does the anodic current at each candidate change cycle1 vs cycle10?
print("\nCurrent at candidate potentials, cycle 1 vs cycle 10 (anodic-going):")
V10,I10 = scans[10]; (_,_),(Vu10,Iu10)=branches(V10,I10)
o10=np.argsort(Vu10); Vs10,Is10=Vu10[o10],Iu10[o10]
for Ehold in [0.5,0.6,0.7,0.8,0.85,0.9]:
    a=Is[np.argmin(np.abs(Vs-Ehold))]; b=Is10[np.argmin(np.abs(Vs10-Ehold))]
    print(f"  E={Ehold:+.2f} V  cyc1={a:7.1f}  cyc10={b:7.1f}  ({100*(b-a)/a:+.0f}%)")
print("\nsaved cv_analysis.png")
