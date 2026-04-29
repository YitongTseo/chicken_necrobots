"""Verify proposed crop across multiple timestamps."""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
VIDEO = HERE / "actuation_currents_data" / "2026.4.16-decel-2soakANDelectropolymeization.mov"
OUT = HERE / "find_crop_2soak.png"

CROP = dict(y0=690, y1=790, x0=1175, x1=1450)

cap = cv2.VideoCapture(str(VIDEO))
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
fracs = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]

fig, axes = plt.subplots(len(fracs), 1, figsize=(8, 1.1*len(fracs)))
for ax, frac in zip(axes, fracs):
    cap.set(cv2.CAP_PROP_POS_MSEC, dur*frac*1000)
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    c = CROP
    ax.imshow(rgb[c['y0']:c['y1'], c['x0']:c['x1']])
    ax.set_title(f"t = {dur*frac:.0f} s", fontsize=9)
    ax.axis('off')
plt.suptitle(f"Proposed crop: y={CROP['y0']}–{CROP['y1']}, x={CROP['x0']}–{CROP['x1']}",
             fontsize=10, y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
cap.release()
print(f"Saved → {OUT}")
