"""Train a small CNN on full XX.XXX display crops.

Input  : RGB crop padded+resized to 96x280
Output : 5 heads × 11 classes (digits 0–9 + "blank" = 10) — leading slot is
         blank for 4-digit readings.
Labels are read from digit_crops/to_label/*.png filenames:
    {digits}_{tag}_{idx}.png   (digits is 4 or 5 chars)
"""
from pathlib import Path
import re
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).parent
LABEL_DIR = HERE / "digit_crops" / "to_label"
OUT_WEIGHTS = HERE / "display_cnn.pt"

IMG_H, IMG_W = 96, 320  # chosen so 4×pool → 6×20, pools cleanly to 1×5
N_SLOTS = 5
N_CLASSES = 11  # 0-9 + blank(=10)
BLANK = 10

BATCH = 32
EPOCHS = 60
LR = 1e-3
SEED = 0
VAL_FRAC = 0.15

FNAME_RE = re.compile(r"^(?P<label>\d+)_(?P<tag>.+)_(?P<idx>\d+)$")


def pad_resize(img, out_h=IMG_H, out_w=IMG_W):
    h, w = img.shape[:2]
    s = min(out_w / w, out_h / h)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def label_to_slots(label):
    """'7623' -> [10, 7, 6, 2, 3]   '11742' -> [1, 1, 7, 4, 2]"""
    digs = [int(c) for c in label]
    if len(digs) > N_SLOTS:
        raise ValueError(label)
    return [BLANK] * (N_SLOTS - len(digs)) + digs


class DisplayDataset(Dataset):
    def __init__(self, files, augment=False):
        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        m = FNAME_RE.match(p.stem)
        label = m.group("label")
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pad_resize(img)
        if self.augment:
            # brightness/contrast jitter
            a = 1.0 + (random.random() - 0.5) * 0.4
            b = (random.random() - 0.5) * 30
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            # spatial augmentation: scale + rotate + translate via affine
            # all computed so that no digit content leaves the frame
            scale = 1.0 + (random.random() - 0.5) * 0.1   # 0.95 – 1.05
            angle = (random.random() - 0.5) * 2.0          # ±1 degree
            cx, cy = IMG_W / 2, IMG_H / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            margin_x = max(1, int(IMG_W * 0.05))  # ~16 px
            margin_y = max(1, int(IMG_H * 0.05))  # ~5 px
            tx = random.randint(-margin_x, margin_x)
            ty = random.randint(-margin_y, margin_y)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (IMG_W, IMG_H), borderValue=(0, 0, 0))
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(label_to_slots(label), dtype=torch.long)
        return x, y


class DisplayCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.backbone = nn.Sequential(
            block(3, 32),   # 48x140
            block(32, 64),  # 24x70
            block(64, 128), # 12x35
            block(128, 128),# 6x17
        )
        # Backbone output: 128 × 6 × 20. Pool (6,4) → 1 × 5 exactly.
        self.head = nn.Sequential(
            nn.AvgPool2d(kernel_size=(6, 4)),
            nn.Flatten(start_dim=2),
        )
        self.fc = nn.Conv1d(128, N_CLASSES, 1)    # -> 11 x 5

    def forward(self, x):
        f = self.backbone(x)
        f = self.head(f)             # (B, 128, 5)
        logits = self.fc(f)          # (B, 11, 5)
        return logits.transpose(1, 2)  # (B, 5, 11)


def collect_files():
    files = []
    for p in sorted(LABEL_DIR.glob("*.png")):
        m = FNAME_RE.match(p.stem)
        if m and m.group("label").isdigit():
            files.append(p)
    return files


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files = collect_files()
    random.shuffle(files)
    n_val = int(len(files) * VAL_FRAC)
    val_files, tr_files = files[:n_val], files[n_val:]
    print(f"Train: {len(tr_files)}   Val: {len(val_files)}")

    tr = DataLoader(DisplayDataset(tr_files, augment=True),
                    batch_size=BATCH, shuffle=True, num_workers=0)
    va = DataLoader(DisplayDataset(val_files, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = DisplayCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, 5, 11)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(-1)
            correct += (pred == y).all(dim=1).sum().item()
            tot += x.size(0)
        tr_loss = loss_sum / tot; tr_acc = correct / tot

        model.eval()
        vtot, vcorr, vslot = 0, 0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                vcorr += (pred == y).all(dim=1).sum().item()
                vslot += (pred == y).sum().item()
                vtot += x.size(0)
        vacc = vcorr / max(vtot, 1); vsacc = vslot / max(vtot * N_SLOTS, 1)
        sched.step()
        print(f"ep {ep:3d}  tr_loss {tr_loss:.4f}  tr_acc {tr_acc:.3f}  "
              f"val_full {vacc:.3f}  val_slot {vsacc:.3f}")

        if vacc > best_acc:
            best_acc = vacc
            torch.save({"model": model.state_dict(),
                        "img_h": IMG_H, "img_w": IMG_W,
                        "n_slots": N_SLOTS, "n_classes": N_CLASSES,
                        "blank": BLANK}, OUT_WEIGHTS)

    print(f"\nBest val full-display accuracy: {best_acc:.3f}")
    print(f"Saved → {OUT_WEIGHTS}")


def predict_voltage(model, device, crop_bgr, n_digits, v_min=None, v_max=None):
    """crop_bgr → voltage float (or None). Used from the notebook."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = pad_resize(img)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(x.to(device))
    pred = logits.argmax(-1).cpu().numpy()[0]  # (5,)
    # Keep last n_digits slots; earlier slots must be BLANK
    digs = pred[-n_digits:]
    if any(d == BLANK for d in digs):
        return None
    s = "".join(str(d) for d in digs)
    v = float(f"{int(s[:-3] or 0)}.{s[-3:]}")
    if v_min is not None and not (v_min <= v <= v_max):
        return None
    return v


if __name__ == "__main__":
    train()
