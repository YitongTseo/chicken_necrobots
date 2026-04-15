"""Quick-label tkinter UI for digit_crops/to_label/.

Keys:
  Enter     save label (renames file) + next
  →  or  n  next (no save)
  ←  or  p  previous
  d         mark BAD_ (unreadable), advance
  t         jump to next suspicious (X-label or BAD_)
  s         save labels.csv snapshot
  q / Esc   quit

Filename format preserved: {label}_{tag}_{idx:03d}.png
"""
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

HERE = Path(__file__).parent
LABEL_DIR = HERE / "digit_crops" / "to_label"

# Expected n_digits per tag (for validation feedback, not enforcement)
N_DIGITS = {"baseline": 4, "decell_1soak": 5, "decell_3soak": 4, "decell_fresh": 4}

FNAME_RE = re.compile(r"^(?P<label>[^_]+)_(?P<tag>.+)_(?P<idx>\d+)$")


def parse(path):
    m = FNAME_RE.match(path.stem)
    if not m:
        return None, None, None
    return m.group("label"), m.group("tag"), m.group("idx")


def is_suspicious(path):
    label, tag, _ = parse(path)
    if label is None:
        return True
    if label.startswith("BAD"):
        return True
    if "X" in label:
        return True
    n = N_DIGITS.get(tag)
    if n and (not label.isdigit() or len(label) != n):
        return True
    return False


class Labeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Display Crop Labeler")
        self.geometry("900x520")
        self.files = sorted(LABEL_DIR.glob("*.png"))
        if not self.files:
            raise SystemExit(f"No PNGs in {LABEL_DIR}")
        self.i = 0

        self.img_label = tk.Label(self, bg="black")
        self.img_label.pack(pady=10)

        self.meta = tk.Label(self, text="", font=("Menlo", 12))
        self.meta.pack()

        row = tk.Frame(self)
        row.pack(pady=8)
        tk.Label(row, text="Label:", font=("Menlo", 14)).pack(side=tk.LEFT)
        self.entry = tk.Entry(row, font=("Menlo", 18), width=10, justify="center")
        self.entry.pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(self, text="", fg="gray", font=("Menlo", 10))
        self.status.pack(pady=4)

        self.progress = ttk.Progressbar(self, length=600, mode="determinate")
        self.progress.pack(pady=4)

        self.bind("<Return>", self.save_and_next)
        self.bind("<Right>", lambda e: self.jump(+1))
        self.bind("n", lambda e: self.jump(+1))
        self.bind("<Left>", lambda e: self.jump(-1))
        self.bind("p", lambda e: self.jump(-1))
        self.bind("d", self.mark_bad)
        self.bind("t", self.next_suspicious)
        self.bind("s", lambda e: self.snapshot_csv())
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("q", lambda e: self.destroy())

        self.show()

    def show(self):
        path = self.files[self.i]
        img = Image.open(path)
        scale = min(800 / img.width, 320 / img.height, 6.0)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.NEAREST)
        self.tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tkimg)

        label, tag, idx = parse(path)
        nd = N_DIGITS.get(tag, "?")
        warn = "  ⚠ SUSPICIOUS" if is_suspicious(path) else ""
        self.meta.config(text=f"[{self.i+1}/{len(self.files)}]  {path.name}   "
                              f"tag={tag}  expects {nd} digits{warn}")
        self.entry.delete(0, tk.END)
        self.entry.insert(0, label or "")
        self.entry.focus_set()
        self.entry.select_range(0, tk.END)
        self.progress["maximum"] = len(self.files)
        self.progress["value"] = self.i + 1

    def save_and_next(self, event=None):
        new_label = self.entry.get().strip()
        path = self.files[self.i]
        label, tag, idx = parse(path)
        if tag is None or idx is None:
            self.status.config(text=f"Can't parse {path.name}; skipping.", fg="red")
            self.jump(+1); return
        if not new_label:
            self.status.config(text="Empty label; skipped.", fg="orange")
            self.jump(+1); return
        new_path = path.with_name(f"{new_label}_{tag}_{idx}.png")
        if new_path != path:
            if new_path.exists():
                self.status.config(text=f"{new_path.name} exists; not renamed.", fg="red")
                self.jump(+1); return
            path.rename(new_path)
            self.files[self.i] = new_path
            self.status.config(text=f"Saved {new_path.name}", fg="green")
        else:
            self.status.config(text="No change.", fg="gray")
        self.jump(+1)

    def mark_bad(self, event=None):
        path = self.files[self.i]
        label, tag, idx = parse(path)
        if tag is None:
            self.jump(+1); return
        new_path = path.with_name(f"BAD_{tag}_{idx}.png")
        if not new_path.exists():
            path.rename(new_path)
            self.files[self.i] = new_path
            self.status.config(text=f"Marked BAD: {new_path.name}", fg="orange")
        self.jump(+1)

    def jump(self, delta):
        self.i = max(0, min(len(self.files) - 1, self.i + delta))
        self.show()

    def next_suspicious(self, event=None):
        for j in range(self.i + 1, len(self.files)):
            if is_suspicious(self.files[j]):
                self.i = j; self.show(); return
        self.status.config(text="No more suspicious files.", fg="gray")

    def snapshot_csv(self):
        out = HERE / "digit_crops" / "labels.csv"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            f.write("filename,label,tag\n")
            for p in self.files:
                label, tag, _ = parse(p)
                f.write(f"{p.name},{label or ''},{tag or ''}\n")
        self.status.config(text=f"Wrote {out}", fg="blue")


if __name__ == "__main__":
    Labeler().mainloop()
