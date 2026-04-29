"""One-off: harvest crops for decell_2soak only."""
import easyocr
from harvest_crops import EXPERIMENTS, harvest

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
harvest(reader, "decell_2soak", EXPERIMENTS["decell_2soak"], n=80)
