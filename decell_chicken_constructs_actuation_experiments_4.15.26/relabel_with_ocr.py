"""Re-run EasyOCR on every crop in digit_crops/to_label/ and rename the file
with the best guess, so you can click-through in label_ui.py.

Only renames; does not delete. Falls back to raw OCR digits (no range check)
if the strict range match fails, so you always get something to confirm.
"""
from pathlib import Path
import re
import cv2
import easyocr

HERE = Path(__file__).parent
LABEL_DIR = HERE / "digit_crops" / "to_label"
N_DIGITS = {"baseline": 4, "decell_1soak": 5, "decell_3soak": 4, "decell_fresh": 4}
V_RANGE = {"baseline": (6.5, 9.0), "decell_1soak": (9.0, 14.0),
           "decell_3soak": (1.0, 6.0), "decell_fresh": (1.0, 6.0)}
FNAME_RE = re.compile(r"^(?P<label>[^_]+)_(?P<tag>.+)_(?P<idx>\d+)$")


def parse(path):
    m = FNAME_RE.match(path.stem)
    if not m:
        return None, None, None
    return m.group("label"), m.group("tag"), m.group("idx")


def assemble(d):
    i = d[:-3] or "0"
    return float(f"{i}.{d[-3:]}")


def guess_label(reader, img, n_digits, v_min, v_max):
    res = reader.readtext(img, detail=1, allowlist="0123456789.")
    if not res:
        return None
    res = sorted(res, key=lambda r: -r[2])
    raw_candidates = []
    for r in res:
        s = r[1].replace(".", "").strip()
        if s.isdigit():
            raw_candidates.append(s)
    if not raw_candidates:
        return None
    # 1) strict: correct length + in range
    for s in raw_candidates:
        if len(s) == n_digits and v_min <= assemble(s) <= v_max:
            return s
    # 2) off-by-one prepend
    for s in raw_candidates:
        if len(s) == n_digits - 1:
            for pref in ("0", "1"):
                c = pref + s
                if v_min <= assemble(c) <= v_max:
                    return c
    # 3) off-by-one truncate
    for s in raw_candidates:
        if len(s) == n_digits + 1 and v_min <= assemble(s[1:]) <= v_max:
            return s[1:]
    # 4) fallback: best-confidence raw digits, trimmed/padded to n_digits
    s = raw_candidates[0]
    if len(s) > n_digits:
        s = s[-n_digits:]
    elif len(s) < n_digits:
        s = s.rjust(n_digits, "0")
    return s


if __name__ == "__main__":
    print("Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    files = sorted(LABEL_DIR.glob("*.png"))
    print(f"Re-labeling {len(files)} files...")
    renamed = skipped = 0
    for p in files:
        label, tag, idx = parse(p)
        if tag is None or tag not in N_DIGITS:
            skipped += 1
            continue
        img = cv2.imread(str(p))
        guess = guess_label(reader, img, N_DIGITS[tag], *V_RANGE[tag])
        if guess is None:
            guess = "X" * N_DIGITS[tag]
        new = p.with_name(f"{guess}_{tag}_{idx}.png")
        if new != p and not new.exists():
            p.rename(new)
            renamed += 1
    print(f"Renamed: {renamed}   Skipped: {skipped}   Unchanged: {len(files)-renamed-skipped}")
