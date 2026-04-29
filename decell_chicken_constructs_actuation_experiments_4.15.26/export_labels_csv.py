"""Write digit_crops/labels.csv from current filenames in digit_crops/to_label/.

Columns: filename, label, tag, n_digits, voltage.
Files prefixed BAD_ or containing 'X' in the label are marked invalid=1.
"""
from pathlib import Path
import re
import csv

HERE = Path(__file__).parent
LABEL_DIR = HERE / "digit_crops" / "to_label"
OUT = HERE / "digit_crops" / "labels.csv"
N_DIGITS = {"baseline": 4, "decell_1soak": 5, "decell_3soak": 4,
            "decell_fresh": 4, "decell_2soak": 4, "decell_3soak_dry_rewet": 4}
FNAME_RE = re.compile(r"^(?P<label>[^_]+)_(?P<tag>.+)_(?P<idx>\d+)$")


def voltage(lbl):
    if not lbl.isdigit():
        return ""
    return f"{int(lbl[:-3] or 0)}.{lbl[-3:]}"


rows = []
for p in sorted(LABEL_DIR.glob("*.png")):
    m = FNAME_RE.match(p.stem)
    if not m:
        continue
    label, tag = m.group("label"), m.group("tag")
    nd = N_DIGITS.get(tag, "")
    invalid = int(label.startswith("BAD") or "X" in label
                  or (nd and (not label.isdigit() or len(label) != nd)))
    rows.append([p.name, label, tag, nd, voltage(label) if not invalid else "", invalid])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label", "tag", "n_digits", "voltage_v", "invalid"])
    w.writerows(rows)

n_total = len(rows)
n_invalid = sum(r[-1] for r in rows)
print(f"Wrote {OUT}")
print(f"  Total: {n_total}   Invalid: {n_invalid}   Usable: {n_total - n_invalid}")
by_tag = {}
for r in rows:
    by_tag.setdefault(r[2], [0, 0])
    by_tag[r[2]][0] += 1
    by_tag[r[2]][1] += r[-1]
for tag, (n, bad) in sorted(by_tag.items()):
    print(f"  {tag:14s}  total={n:3d}  invalid={bad}")
