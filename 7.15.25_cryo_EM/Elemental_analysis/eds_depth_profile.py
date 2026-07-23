"""
EDS depth-profile plots for the electro-pol FIB cross-section (electro-pol-FIB_08_SPOT).

Depth order from the surface downward:  EDS07, EDS01, EDS02, EDS03, EDS04, EDS05, EDS06.
Composition = Wt% read from each EDS0x spectrum's Tru-Q table.
Pixel Y positions of each spot were measured directly from electro-pol-FIB_08_SPOT.png.

Outputs (written next to this script):
    v1_depth_profile_lines.png    - line graph, depth on Y (surface at top), wt% on X
    v2_overlay_on_image.png       - profile overlaid on the SEM image at true pixel depths
    v3_depth_um_lines.png         - same as v1 but Y axis calibrated to microns (scale bar)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# ----------------------------------------------------------------------------
# Data.  Order = physical depth from the surface down.
spots = ["EDS07", "EDS01", "EDS02", "EDS03", "EDS04", "EDS05", "EDS06"]

# wt% per element; missing element = 0.0
elements = ["C", "Cl", "S", "O", "Ga", "P"]
comp = {
    #         C     Cl    S     O    Ga    P
    "EDS07": [24.4, 0.0,  6.3,  2.8, 60.6, 0.0],
    "EDS01": [72.4, 9.2,  11.9, 4.5, 1.8,  0.3],
    "EDS02": [58.7, 24.4, 11.8, 2.6, 2.1,  0.5],
    "EDS03": [41.9, 41.2, 10.1, 3.8, 2.5,  0.4],
    "EDS04": [45.1, 38.6, 9.5,  3.9, 2.5,  0.4],
    "EDS05": [46.9, 32.5, 11.2, 5.8, 2.7,  0.8],
    "EDS06": [44.9, 35.4, 13.4, 3.9, 2.4,  0.0],
}

# Measured pixel Y (and X) of each spot crosshair in electro-pol-FIB_08_SPOT.png
# (re-measured; EDS01-06 crosshairs verified against the image).
pix_y = {"EDS07": 149, "EDS01": 229, "EDS02": 263, "EDS03": 301,
         "EDS04": 330, "EDS05": 357, "EDS06": 386}
pix_x = {"EDS07": 356, "EDS01": 350, "EDS02": 351, "EDS03": 353,
         "EDS04": 355, "EDS05": 353, "EDS06": 354}

# EDS07 is a floating piece above the surface, so it is NOT part of the depth
# stack.  Depth zero = EDS01 (the true film surface); EDS07 is excluded from the
# profiles below.
spots_depth = ["EDS01", "EDS02", "EDS03", "EDS04", "EDS05", "EDS06"]
SURF = "EDS01"

# Scale bar: 182 px = 5 um  ->  um per pixel
UM_PER_PX = 5.0 / 182.0

# FIB stage tilt. The milled cross-section face is viewed foreshortened, so the
# true depth into the sample = apparent (in-image) depth / sin(tilt).
# 52 deg is the standard Thermo/FEI dual-beam geometry; change this if yours differs.
TILT_DEG = 52.0
TILT_FACTOR = 1.0 / np.sin(np.radians(TILT_DEG))   # ~1.27 for 52 deg

# Validated categorical palette (dataviz skill), one fixed hue per element.
color = {
    "C":  "#2a78d6",  # blue
    "Cl": "#008300",  # green
    "S":  "#e87ba4",  # magenta
    "O":  "#eda100",  # yellow
    "Ga": "#1baf7a",  # aqua
    "P":  "#eb6834",  # orange
}

INK      = "#0b0b0b"
SECOND   = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
SURFACE  = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

# Depth series: EDS01 (surface) -> EDS06, floating EDS07 dropped.
comp_arr   = np.array([comp[s] for s in spots_depth])  # (6 spots, 6 elements)
y_index    = np.arange(len(spots_depth))               # even spacing 0..5
depth_um   = np.array([(pix_y[s] - pix_y[SURF]) * UM_PER_PX for s in spots_depth])  # apparent
depth_true = depth_um * TILT_FACTOR                    # tilt-corrected true depth


# ============================================================================
# VERSION 1 - line graph: depth on Y (surface top), wt% on X, one line/element
# ============================================================================
def v1_lines():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for j, el in enumerate(elements):
        ax.plot(comp_arr[:, j], y_index, marker="o", ms=7, lw=2,
                color=color[el], label=el, zorder=3)
    # direct-label only the two high-amplitude, unambiguous lines; legend covers the rest
    for el, dy in (("C", 6), ("Ga", 6)):
        j = elements.index(el)
        ax.annotate(el, (comp_arr[0, j], y_index[0]),
                    xytext=(4, dy), textcoords="offset points",
                    color=color[el], fontsize=11, fontweight="bold")

    ax.set_ylim(len(spots_depth) - 0.5, -0.7)          # invert: surface (EDS01) on top
    ax.set_yticks(y_index)
    ax.set_yticklabels([f"{s}" for s in spots_depth])
    ax.set_ylabel("EDS spot  (surface at top, deeper below)", fontsize=11, color=SECOND)
    ax.set_xlabel("Composition  (wt %)", fontsize=11, color=SECOND)
    ax.set_xlim(-2, 80)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=SECOND)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # secondary right axis: tilt-corrected true depth in um
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_index)
    ax2.set_yticklabels([f"{d:.1f}" for d in depth_true])
    ax2.set_ylabel(f"true depth from surface (µm, ÷sin {TILT_DEG:.0f}°)",
                   fontsize=10, color=MUTED)
    ax2.tick_params(colors=MUTED)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)

    ax.set_title("EDS depth profile — electro-pol FIB cross-section",
                 fontsize=13, color=INK, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig("v1_depth_profile_lines.png", dpi=200)
    plt.close(fig)


# ============================================================================
# VERSION 2 - SEM image (left) + depth profile (right), the two tied together by
# dashed connectors from each crosshair to its depth row.  Elements shown:
# C, S, O, P (Cl and Ga dropped).  True-depth axis on the right.  EDS07 excluded.
# ============================================================================
def v2_overlay():
    from matplotlib.patches import ConnectionPatch
    els2 = ["C", "S", "O", "P"]                         # Cl and Ga removed

    img = Image.open("electro-pol-FIB_08_SPOT.png").convert("RGB")
    W, H = img.size
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.04)

    # --- left: the SEM section ------------------------------------------------
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img, extent=[0, W, H, 0])
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.axis("off")
    # subtle ring on each spot so the connector origin is unambiguous
    for s in spots_depth:
        ax_img.plot(pix_x[s], pix_y[s], marker="o", ms=9, mfc="none",
                    mec="#f5d020", mew=1.3, zorder=5)

    # --- right: the depth profile --------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for el in els2:
        j = elements.index(el)
        ax.plot(comp_arr[:, j], depth_true, marker="o", ms=6, lw=1.9,
                color=color[el], label=el, zorder=3)
    for el in ("C", "S"):                              # direct-label the two clear lines
        j = elements.index(el)
        ax.annotate(el, (comp_arr[0, j], depth_true[0]),
                    xytext=(5, -2), textcoords="offset points",
                    color=color[el], fontsize=11, fontweight="bold", zorder=4)

    ax.set_ylim(depth_true.max() + 0.4, -0.4)          # surface (EDS01) on top
    ax.set_xlim(-2, 80)
    ax.set_xlabel("composition (wt %)", fontsize=10, color=SECOND)
    # true-depth axis on the RIGHT
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"true depth from surface (µm, ÷sin {TILT_DEG:.0f}°)",
                  fontsize=10, color=SECOND)
    ax.set_yticks(depth_true)
    ax.set_yticklabels([f"{d:.1f}" for d in depth_true])
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=SECOND)
    for sp in ("top", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)

    # --- dashed connectors: crosshair (image) -> depth row (profile baseline) --
    for i, s in enumerate(spots_depth):
        con = ConnectionPatch(
            xyA=(pix_x[s], pix_y[s]), coordsA=ax_img.transData,
            xyB=(-2, depth_true[i]), coordsB=ax.transData,
            color=MUTED, lw=0.8, ls=(0, (4, 3)), alpha=0.8, zorder=1)
        fig.add_artist(con)

    fig.suptitle("EDS composition vs. true depth — keyed to the FIB cross-section "
                 "(EDS01 = surface; floating EDS07 excluded)",
                 fontsize=12, color=INK, fontweight="bold", y=0.99)
    fig.savefig("v2_overlay_on_image.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# VERSION 3 - Y axis is the TILT-CORRECTED true depth (µm), so vertical spacing
# reflects the real depth into the sample.
# ============================================================================
def v3_depth_um():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for j, el in enumerate(elements):
        ax.plot(comp_arr[:, j], depth_true, marker="o", ms=7, lw=2,
                color=color[el], label=el, zorder=3)
    for el, dy in (("C", 8), ("Ga", 8)):
        j = elements.index(el)
        ax.annotate(el, (comp_arr[0, j], depth_true[0]),
                    xytext=(4, dy), textcoords="offset points",
                    color=color[el], fontsize=11, fontweight="bold")

    ax.set_ylim(depth_true.max() + 0.5, -0.5)          # surface on top
    ax.set_ylabel(f"true depth from surface (µm, tilt-corrected ÷sin {TILT_DEG:.0f}°)",
                  fontsize=11, color=SECOND)
    ax.set_xlabel("Composition  (wt %)", fontsize=11, color=SECOND)
    ax.set_xlim(-2, 80)
    # secondary spot-name labels on the right edge, at each real depth
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(depth_true)
    ax2.set_yticklabels(spots_depth)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=SECOND)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.set_title("EDS depth profile (true depth) — FIB cross-section",
                 fontsize=13, color=INK, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig("v3_true_depth_lines.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    v1_lines()
    v2_overlay()
    v3_depth_um()
    print("wrote v1_depth_profile_lines.png, v2_overlay_on_image.png, v3_true_depth_lines.png")
