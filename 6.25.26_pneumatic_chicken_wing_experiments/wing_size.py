#!/usr/bin/env python3
"""
Measure each wing's SIZE from its rest frame, so deflection can be normalized.

For every configured video we take a denoised "wing at rest" still (median of
the first REST_FRAMES frames), segment the tissue away from the graph-paper
background (warm colour in LAB space; see wd.segment_wing), and measure:
  * area_mm2          -- segmented tissue area
  * sqrt_area_mm      -- sqrt(area): a pose-robust "effective size" (the value
                         the benchmark normalizes deflection by)
  * caliper_length_mm -- max base->tip caliper ("overall length"), for the
                         wing-length distribution plot

Calibration (px/mm) is reused from deflection_config.GRID_OVERRIDE -- the same
scale the tracker used -- so no clicking is needed for the automatic pass.

Outputs per video:
  deflection_analysis/wing_size/<stem>.json   sizes + calibration
  deflection_analysis/wing_size/<stem>.png    verification: contour + caliper

Modes:
  default    : fully automatic segmentation (headless, no clicks).
  --manual   : click a POLYGON around the wing (bulletproof fallback for clips
               the auto pass under-segments -- e.g. wings wrapped in opaque white
               actuator rings). ENTER closes the polygon, R resets, ESC aborts.
  --proximal : click TWO points -- the SHOULDER (where the wing meets the mount)
               then the ELBOW (first joint) -- to record the humerus segment
               length. This is a pose-invariant scale: it doesn't change with how
               the wing is folded at the elbow/wrist, unlike the full caliper
               length. Adds proximal_length_mm to each clip's existing JSON; the
               notebook normalizes by it. Run the automatic/area pass FIRST.

Re-running skips finished videos; pass --redo (optionally with --video) to redo.

Examples:
  python wing_size.py                                    # all, automatic area
  python wing_size.py --video "<path>" --redo            # redo one clip (auto)
  python wing_size.py --video "<path>" --redo --manual   # hand-trace a hard clip
  python wing_size.py --proximal                         # 2-click shoulder->elbow, all clips
  python wing_size.py --video "<path>" --proximal --redo # re-measure one clip's humerus
"""
import os
import csv
import json
import argparse

import cv2
import numpy as np

import deflection_config as cfg
import wing_deflection as wd


def _px_per_mm(rel_path):
    """Reuse the tracker's calibration; None if this clip has no grid override."""
    pps = cfg.GRID_OVERRIDE.get(rel_path)
    return (pps / cfg.GRID_MM_PER_SQUARE) if pps else None


# --------------------------------------------------------------------------
# interactive polygon fallback
# --------------------------------------------------------------------------
def click_polygon(frame):
    """Click wing-outline vertices; ENTER closes & fills. Returns a 0/255 mask."""
    win = "MANUAL wing outline  (click vertices, ENTER=close, R=reset, ESC=abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    pts = []

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.setMouseCallback(win, on_mouse)
    mask = None
    while True:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            if i:
                cv2.line(disp, pts[i - 1], p, (0, 0, 255), 2)
        if len(pts) > 2:
            cv2.line(disp, pts[-1], pts[0], (0, 200, 255), 1)
        cv2.putText(disp, f"{len(pts)} pts  ENTER=close R=reset ESC=abort",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) >= 3:
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
            break
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k == 27:
            break
    cv2.destroyWindow(win)
    return mask


def _seg_from_mask(mask):
    """Wrap a hand-drawn mask into the same dict shape as wd.segment_wing()."""
    cal_px, cal_pts = wd._caliper_length_px(mask)
    return dict(mask=mask, area_px=float(cv2.countNonZero(mask)),
                caliper_px=cal_px, caliper_pts=cal_pts, ok=True)


def click_two_points(frame, prompt):
    """Capture exactly two clicks; returns [(x,y),(x,y)] or None if aborted."""
    win = prompt + "  (click 2 pts, ENTER=done, R=reset, ESC=abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    pts = []

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.setMouseCallback(win, on_mouse)
    out = None
    while True:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 6, (255, 0, 0), -1)
            cv2.putText(disp, ["shoulder", "elbow"][i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (255, 0, 0), 2)
        cv2.putText(disp, f"{prompt}  [{len(pts)}/2]", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 2:
            out = pts
            break
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k == 27:
            break
    cv2.destroyWindow(win)
    return out


def process_proximal(group, abs_path, rel_path, args):
    """Add proximal (shoulder->elbow) length to a clip's existing wing-size JSON."""
    stem = cfg.result_stem(rel_path)
    json_path = os.path.join(cfg.WING_SIZE_DIR, stem + ".json")
    png_path = os.path.join(cfg.WING_SIZE_DIR, stem + ".png")

    if not os.path.exists(json_path):
        print(f"  [skip] {rel_path}: no wing-size JSON yet; run the area pass first")
        return
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("proximal_length_mm") is not None and not args.redo:
        print(f"  [skip] {rel_path} (proximal already measured)")
        return
    if not os.path.exists(abs_path):
        print(f"  [MISSING] {rel_path}")
        return

    print(f"\n=== {group}  |  {rel_path} ===")
    rest = wd.rest_frame_median(abs_path, cfg.REST_FRAMES)
    pts = click_two_points(rest, "SHOULDER then ELBOW")
    if pts is None:
        print("  [skip] proximal measurement aborted")
        return

    px_per_mm = meta["px_per_mm"]
    length_mm = float(np.hypot(pts[1][0] - pts[0][0],
                               pts[1][1] - pts[0][1]) / px_per_mm)
    meta["proximal_pts"] = [list(map(int, p)) for p in pts]
    meta["proximal_length_mm"] = length_mm
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    proximal (humerus) length {length_mm:.1f} mm")

    # annotate the existing overlay with the shoulder->elbow segment (blue)
    base = cv2.imread(png_path) if os.path.exists(png_path) else rest.copy()
    cv2.line(base, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 3)
    for p, lab in zip(pts, ("shoulder", "elbow")):
        cv2.circle(base, tuple(p), 6, (255, 0, 0), -1)
        cv2.putText(base, lab, (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    txt = f"proximal len = {length_mm:.0f} mm"
    cv2.putText(base, txt, (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(base, txt, (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(png_path, base)


# --------------------------------------------------------------------------
# per-video driver
# --------------------------------------------------------------------------
def process(group, abs_path, rel_path, args):
    stem = cfg.result_stem(rel_path)
    json_path = os.path.join(cfg.WING_SIZE_DIR, stem + ".json")
    png_path = os.path.join(cfg.WING_SIZE_DIR, stem + ".png")

    if os.path.exists(json_path) and not args.redo:
        print(f"  [skip] {rel_path} (already done)")
        return
    if not os.path.exists(abs_path):
        print(f"  [MISSING] {rel_path}")
        return

    px_per_mm = _px_per_mm(rel_path)
    if not px_per_mm:
        print(f"  [skip] {rel_path}: no GRID_OVERRIDE; add one to "
              f"deflection_config to set the px/mm scale")
        return

    print(f"\n=== {group}  |  {rel_path} ===")
    rest = wd.rest_frame_median(abs_path, cfg.REST_FRAMES)

    if args.manual:
        mask = click_polygon(rest)
        if mask is None:
            print("  [skip] manual outline aborted")
            return
        seg, mode = _seg_from_mask(mask), "manual"
    else:
        seg, mode = wd.segment_wing(rest), "auto"
        if not seg["ok"]:
            print("  [WARN] auto segmentation found no plausible wing; "
                  "re-run with --manual")

    sizes = wd.wing_size_metrics(seg, px_per_mm)
    print(f"    sqrt(area) {sizes['sqrt_area_mm']:.1f} mm | "
          f"length {sizes['caliper_length_mm']:.1f} mm | "
          f"area {sizes['area_mm2']:.0f} mm^2  ({mode})")

    os.makedirs(cfg.WING_SIZE_DIR, exist_ok=True)
    cv2.imwrite(png_path, wd.draw_wing_overlay(rest, seg, sizes))
    meta = dict(
        group=group, rel_path=rel_path, mode=mode,
        px_per_mm=float(px_per_mm),
        px_per_square=float(px_per_mm * cfg.GRID_MM_PER_SQUARE),
        rest_frames=cfg.REST_FRAMES,
        seg_ok=bool(seg["ok"]),
        area_px=float(seg["area_px"]),
        area_mm2=float(sizes["area_mm2"]),
        sqrt_area_mm=float(sizes["sqrt_area_mm"]),
        caliper_length_mm=float(sizes["caliper_length_mm"]),
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    overlay -> {os.path.relpath(png_path, cfg.ROOT)}")


def write_summary():
    """Roll every wing_size/*.json into one CSV for quick review."""
    rows = []
    for fn in sorted(os.listdir(cfg.WING_SIZE_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(cfg.WING_SIZE_DIR, fn)) as f:
                rows.append(json.load(f))
    if not rows:
        return
    cols = ["group", "rel_path", "mode", "seg_ok", "sqrt_area_mm",
            "caliper_length_mm", "proximal_length_mm", "area_mm2", "px_per_mm"]
    out = os.path.join(cfg.WING_SIZE_DIR, "wing_sizes.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsummary -> {os.path.relpath(out, cfg.ROOT)}  ({len(rows)} videos)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="process only this relative path")
    ap.add_argument("--redo", action="store_true", help="redo finished videos")
    ap.add_argument("--manual", action="store_true",
                    help="hand-trace the wing polygon (fallback for hard clips)")
    ap.add_argument("--proximal", action="store_true",
                    help="2-click shoulder->elbow (humerus) length, pose-invariant; "
                         "run the area pass first")
    args = ap.parse_args()

    os.makedirs(cfg.WING_SIZE_DIR, exist_ok=True)
    todo = [(g, a, r) for (g, a, r) in cfg.all_videos()
            if (args.video is None or r == args.video)]
    if not todo:
        print("no matching videos (check --video path)")
        return
    print(f"{len(todo)} video(s) to consider")
    driver = process_proximal if args.proximal else process
    for group, abs_path, rel_path in todo:
        try:
            driver(group, abs_path, rel_path, args)
        except Exception as e:
            print(f"  [ERROR] {rel_path}: {e}")
    write_summary()
    print("\ndone. Review masks in deflection_analysis/wing_size/*.png; "
          "re-run weak ones with --manual, then run the analysis notebook.")


if __name__ == "__main__":
    main()
