"""Architecture diagram for the display CNN."""
from pathlib import Path
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from train_display_cnn import DisplayCNN, pad_resize, BLANK, N_SLOTS, N_CLASSES

HERE = Path(__file__).parent
LABEL_DIR = HERE / "digit_crops" / "to_label"
OUT = HERE / "display_cnn_architecture.png"
random.seed(2)

# ── Samples, one per experiment ───────────────────────────────────────────
tags = ["baseline", "decell_1soak", "decell_3soak", "decell_fresh"]
samples = []
for tag in tags:
    p = random.choice(sorted(LABEL_DIR.glob(f"*_{tag}_*.png")))
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    samples.append((tag, p.stem.split("_")[0], img))

# ── Real forward pass on one sample ───────────────────────────────────────
device = torch.device("cpu")
ckpt = torch.load(HERE / "display_cnn.pt", map_location=device, weights_only=True)
model = DisplayCNN().to(device).eval()
model.load_state_dict(ckpt["model"])
demo_tag, demo_label, demo_img = samples[1]
x = torch.from_numpy(pad_resize(demo_img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
with torch.no_grad():
    probs = torch.softmax(model(x), dim=-1)[0].numpy()  # (5, 11)
pred_slots = probs.argmax(axis=-1)

# ── Figure canvas ─────────────────────────────────────────────────────────
FW, FH = 20, 9
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

BLOCK_COLOR = "#cfe2f3"
BLOCK_EDGE = "#1f4e79"
HEAD_COLOR = "#f4cccc"
HEAD_EDGE = "#990000"
HEATMAP_CMAP = "Blues"

# ── Title ─────────────────────────────────────────────────────────────────
ax.text(FW / 2, 8.55, "Display CNN  —  5-digit 7-segment reader",
        ha="center", fontsize=18, fontweight="bold")
ax.text(FW / 2, 8.15,
        "368 labeled crops · 40 epochs · 98.2% full-display accuracy on held-out validation",
        ha="center", fontsize=11, style="italic", color="#555")

# ── 1) Input stack ────────────────────────────────────────────────────────
input_x = 0.3
input_w = 3.1
ax.text(input_x + input_w / 2, 7.55, "Input crops",
        ha="center", fontsize=12, fontweight="bold")
ax.text(input_x + input_w / 2, 7.22, "padded & resized to 96 × 320",
        ha="center", fontsize=9.5, style="italic", color="#555")
for i, (tag, label, img) in enumerate(samples):
    y0 = 6.5 - i * 1.45
    h_box = 1.0
    disp = cv2.resize(pad_resize(img), (int(input_w * 60), int(h_box * 60)))
    ax.imshow(disp, extent=(input_x, input_x + input_w, y0, y0 + h_box),
              aspect="auto", zorder=2)
    ax.add_patch(Rectangle((input_x, y0), input_w, h_box, fill=False,
                           edgecolor="#333", lw=0.8, zorder=3))
    ax.text(input_x + input_w / 2, y0 - 0.22,
            f"{tag}  ·  reads {int(label[:-3] or 0)}.{label[-3:]} V",
            ha="center", fontsize=9, family="monospace", color="#333")

# Arrow into backbone
ax.add_patch(FancyArrowPatch((input_x + input_w + 0.05, 4.3),
                             (input_x + input_w + 0.75, 4.3),
                             arrowstyle="->", mutation_scale=24, lw=1.8, color="black"))

# ── 2) Backbone header ────────────────────────────────────────────────────
backbone_x0 = input_x + input_w + 0.9
backbone_y = 4.3
block_w = 1.55
gap = 0.28
n_blocks = 4
backbone_total = n_blocks * block_w + (n_blocks - 1) * gap

ax.text(backbone_x0 + backbone_total / 2, 7.55, "CNN backbone",
        ha="center", fontsize=12, fontweight="bold")
ax.text(backbone_x0 + backbone_total / 2, 7.22,
        "per block:  Conv 3×3 (pad 1) → BN → ReLU   ×2   → MaxPool 2×2",
        ha="center", fontsize=9.5, style="italic", color="#555")

blocks = [
    ("block 1",  "32 ch",  "48 × 160"),
    ("block 2",  "64 ch",  "24 × 80"),
    ("block 3", "128 ch",  "12 × 40"),
    ("block 4", "128 ch",  "6 × 20"),
]
heights = [3.8, 3.1, 2.4, 1.8]  # shrinking to hint spatial downsampling

for i, ((name, ch, shape), h) in enumerate(zip(blocks, heights)):
    x0 = backbone_x0 + i * (block_w + gap)
    y0 = backbone_y - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), block_w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.6, facecolor=BLOCK_COLOR, edgecolor=BLOCK_EDGE, zorder=2))
    ax.text(x0 + block_w / 2, y0 + h / 2 + 0.30, name,
            ha="center", va="center", fontsize=11, fontweight="bold", color=BLOCK_EDGE)
    ax.text(x0 + block_w / 2, y0 + h / 2, ch,
            ha="center", va="center", fontsize=10.5, family="monospace")
    ax.text(x0 + block_w / 2, y0 + h / 2 - 0.30, shape,
            ha="center", va="center", fontsize=9.5, family="monospace", color="#333")
    if i < n_blocks - 1:
        x_next = backbone_x0 + (i + 1) * (block_w + gap)
        ax.add_patch(FancyArrowPatch((x0 + block_w, backbone_y),
                                     (x_next, backbone_y),
                                     arrowstyle="->", mutation_scale=14,
                                     lw=1.0, color="#888"))

# Arrow into head
head_start_x = backbone_x0 + backbone_total + 0.1
ax.add_patch(FancyArrowPatch((backbone_x0 + backbone_total, backbone_y),
                             (head_start_x + 0.6, backbone_y),
                             arrowstyle="->", mutation_scale=22, lw=1.6, color="black"))

# ── 3) Head (pool + 1×1 conv) ─────────────────────────────────────────────
head_x0 = head_start_x + 0.65
head_w = 2.3
head_h = 2.6
head_y0 = backbone_y - head_h / 2
ax.add_patch(FancyBboxPatch((head_x0, head_y0), head_w, head_h,
             boxstyle="round,pad=0.02,rounding_size=0.10",
             linewidth=1.6, facecolor=HEAD_COLOR, edgecolor=HEAD_EDGE, zorder=2))
ax.text(head_x0 + head_w / 2, head_y0 + head_h - 0.35, "classifier head",
        ha="center", fontsize=11, fontweight="bold", color=HEAD_EDGE)
ax.text(head_x0 + head_w / 2, head_y0 + head_h - 0.85,
        "AvgPool 6×4", ha="center", fontsize=10)
ax.text(head_x0 + head_w / 2, head_y0 + head_h - 1.20,
        "→ 128 × 1 × 5", ha="center", fontsize=9.5, family="monospace", color="#333")
ax.text(head_x0 + head_w / 2, head_y0 + head_h - 1.70,
        "Conv 1×1: 128 → 11", ha="center", fontsize=10)
ax.text(head_x0 + head_w / 2, head_y0 + head_h - 2.05,
        "(per-slot logits)", ha="center", fontsize=9, style="italic", color="#555")

# Arrow into output
ax.add_patch(FancyArrowPatch((head_x0 + head_w, backbone_y),
                             (head_x0 + head_w + 0.6, backbone_y),
                             arrowstyle="->", mutation_scale=22, lw=1.6, color="black"))

# ── 4) Softmax heatmap ────────────────────────────────────────────────────
out_x = head_x0 + head_w + 0.75
out_w = FW - out_x - 0.35
out_h = 5.2
out_y = backbone_y - out_h / 2 + 0.3

ax.text(out_x + out_w / 2, out_y + out_h + 0.55, "Softmax output",
        ha="center", fontsize=12, fontweight="bold")
ax.text(out_x + out_w / 2, out_y + out_h + 0.22,
        "11 classes × 5 slots",
        ha="center", fontsize=9.5, style="italic", color="#555")

display_rows = list(range(N_CLASSES - 1, -1, -1))   # [10 blank, 9, ..., 0]
row_labels = ["_" if k == BLANK else str(k) for k in display_rows]
heat = probs.T[display_rows, :]                      # (11, 5)
cell_w = out_w / 5
cell_h = out_h / 11

ax.imshow(heat, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto",
          extent=(out_x, out_x + out_w, out_y, out_y + out_h), zorder=2)
# Thin gridlines for clarity
for c in range(6):
    ax.plot([out_x + c * cell_w]*2, [out_y, out_y + out_h],
            color="white", lw=0.7, zorder=3)
for r in range(12):
    ax.plot([out_x, out_x + out_w], [out_y + r * cell_h]*2,
            color="white", lw=0.7, zorder=3)
# Outer border
ax.add_patch(Rectangle((out_x, out_y), out_w, out_h, fill=False,
                       edgecolor="#333", lw=1.4, zorder=4))

# Row labels
for r, lbl in enumerate(row_labels):
    y = out_y + out_h - (r + 0.5) * cell_h
    ax.text(out_x - 0.14, y, lbl, ha="right", va="center",
            fontsize=11, family="monospace")
ax.text(out_x - 0.65, out_y + out_h / 2, "class",
        ha="center", va="center", fontsize=11, fontweight="bold",
        rotation=90, color="#333")
# Column labels
for c in range(5):
    cx = out_x + (c + 0.5) * cell_w
    ax.text(cx, out_y - 0.28, f"slot {c+1}", ha="center", va="top",
            fontsize=11, fontweight="bold")

# Argmax highlights
for c in range(5):
    pred = int(pred_slots[c])
    r = display_rows.index(pred)
    rx = out_x + c * cell_w
    ry_top = out_y + out_h - r * cell_h
    ry_bot = ry_top - cell_h
    ax.add_patch(Rectangle((rx, ry_bot), cell_w, cell_h,
                           fill=False, edgecolor="#d62728", lw=2.4, zorder=6))
    # Print the probability inside the winning cell
    ax.text(rx + cell_w / 2, ry_bot + cell_h / 2,
            f"{probs[c, pred]:.2f}", ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color="white" if probs[c, pred] > 0.5 else "#d62728", zorder=7)

# ── 5) Decoded reading ────────────────────────────────────────────────────
decoded_digits = [("_" if d == BLANK else str(int(d))) for d in pred_slots]
decoded = "".join(d for d in decoded_digits if d != "_")
voltage = f"{int(decoded[:-3] or 0)}.{decoded[-3:]}" if decoded else "—"
ax.text(out_x + out_w / 2, out_y - 0.9,
        f"argmax  →  [ {'  '.join(decoded_digits)} ]",
        ha="center", fontsize=11.5, family="monospace")
ax.text(out_x + out_w / 2, out_y - 1.55,
        f"Decoded reading:  {voltage} V",
        ha="center", fontsize=14, fontweight="bold", color="#990000",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff5e6",
                  ec="#990000", lw=1.5))

plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved: {OUT}")
