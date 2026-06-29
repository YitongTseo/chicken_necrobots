#!/usr/bin/env python3
"""
Stage 1 (interactive): track the wing tip in each configured video.

CALIBRATION (px -> mm), default = "ruler" (most reliable):
  Click TWO grid-line intersections a known number of squares apart, then type
  that number in the terminal. Immune to faint/harmonic grid lines.
  Use --calib auto for the quick FFT/autocorr estimate instead (check confidence).

TRACKING, two modes:
  default  : click a TIGHT box on the wing TIP; an adaptive normalized
             cross-correlation tracker follows it. Good for most clips.
  --manual : step through the clip and CLICK THE TIP every N frames
             (--manual-step, default 15). Positions are interpolated.
             Bulletproof fallback for clips the auto tracker can't hold
             (big/fast deflection, bleached tissue, heavy shadow).

Drawing tips that prevent the failures we saw:
  * Tip box: draw it SMALL and TIGHT, centered exactly on the wing tip --
    not the whole wing. A big box tracks the body, not the tip.
  * Ruler clicks: pick two crisp grid intersections far apart (e.g. 10 squares)
    for an accurate scale.

Outputs per video:
  deflection_analysis/tracks/<stem>.csv     per-frame tip track + deflection
  deflection_analysis/tracks/<stem>.json    calibration + metrics + ROIs/keyframes
  deflection_analysis/overlays/<stem>.mp4    verification video (tip + trail)

Re-running skips finished videos; pass --redo (optionally with --video) to redo.

Examples:
  python track_tips.py                                  # all, ruler calib, auto track
  python track_tips.py --calib auto                     # quick auto grid estimate
  python track_tips.py --video "<path>" --redo          # redo one clip
  python track_tips.py --video "<path>" --redo --manual # hand-annotate a hard clip
"""
import os
import csv
import json
import argparse

import cv2
import numpy as np

import deflection_config as cfg
import wing_deflection as wd


# --------------------------------------------------------------------------
# interactive helpers
# --------------------------------------------------------------------------
def open_window(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 720)
    return name


def banner(frame, lines):
    out = frame.copy()
    for i, t in enumerate(lines):
        y = 38 + 34 * i
        cv2.putText(out, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(out, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


def select_roi(window, frame, prompt):
    box = cv2.selectROI(window, banner(frame, [prompt]),
                        showCrosshair=True, fromCenter=False)
    return box  # (x,y,w,h); zeros if cancelled


def click_points(window, frame, n, prompt):
    """Capture exactly `n` mouse clicks; returns list of (x,y) in image coords."""
    pts = []

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))

    cv2.setMouseCallback(window, on_mouse)
    while True:
        disp = banner(frame, [prompt, f"clicked {len(pts)}/{n}  (ENTER=done, R=reset)"])
        for p in pts:
            cv2.circle(disp, p, 6, (0, 0, 255), -1)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (0, 0, 255), 2)
        cv2.imshow(window, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == n:
            break
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k == 27:  # ESC -> abort
            pts.clear()
            break
    cv2.setMouseCallback(window, lambda *a: None)
    return pts


def calibrate(window, frame, mode):
    """Return (px_per_square, info_dict)."""
    if mode == "ruler":
        pts = click_points(window, frame, 2,
                           "RULER: click two grid intersections, then ENTER")
        if len(pts) != 2:
            return None, dict(method="ruler", aborted=True)
        try:
            n_sq = float(input("    how many SQUARES between the two clicks? "))
        except (ValueError, EOFError):
            n_sq = 10.0
            print("    (defaulting to 10 squares)")
        pps = wd.ruler_px_per_square(pts[0], pts[1], n_sq)
        return pps, dict(method="ruler", points=[list(p) for p in pts],
                         n_squares=n_sq, confidence=100.0)
    else:  # auto -- click-free, measures grid-line spacing over the whole frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pps, n = wd.measure_grid(gray)
        # confidence scales with how many clean line-spacings were measured
        conf = 100.0 * min(1.0, n / 50.0) if pps else 0.0
        return pps, dict(method="auto", n_samples=n, confidence=conf)


def manual_annotate(video_path, step):
    """Step through the clip; click the tip each shown frame. Returns keyframes."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_show = list(range(0, n, step))
    if frames_to_show[-1] != n - 1:
        frames_to_show.append(n - 1)   # always annotate the last frame

    win = open_window("MANUAL: click the tip  (ENTER=next, B=back, S=skip, Q=finish)")
    keyframes = {}
    i = 0
    while 0 <= i < len(frames_to_show):
        f = frames_to_show[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            i += 1
            continue
        click = {"pt": keyframes.get(f)}

        def on_mouse(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN:
                click["pt"] = (x, y)

        cv2.setMouseCallback(win, on_mouse)
        while True:
            disp = banner(frame, [f"frame {f}/{n-1}  ({i+1}/{len(frames_to_show)})",
                                  "click TIP; ENTER=next, B=back, S=skip, Q=finish"])
            if click["pt"]:
                cv2.circle(disp, tuple(map(int, click["pt"])), 7, (0, 0, 255), -1)
            cv2.imshow(win, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10):                         # next
                if click["pt"]:
                    keyframes[f] = (float(click["pt"][0]), float(click["pt"][1]))
                i += 1
                break
            if k in (ord("b"), ord("B")):             # back
                if click["pt"]:
                    keyframes[f] = (float(click["pt"][0]), float(click["pt"][1]))
                i = max(0, i - 1)
                break
            if k in (ord("s"), ord("S")):             # skip (no click here)
                i += 1
                break
            if k in (ord("q"), ord("Q")):             # finish
                if click["pt"]:
                    keyframes[f] = (float(click["pt"][0]), float(click["pt"][1]))
                cap.release()
                cv2.destroyWindow(win)
                return keyframes
    cap.release()
    cv2.destroyWindow(win)
    return keyframes


# --------------------------------------------------------------------------
# per-video driver
# --------------------------------------------------------------------------
def first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot read {video_path}")
    return frame


def process(group, abs_path, rel_path, args):
    stem = cfg.result_stem(rel_path)
    json_path = os.path.join(cfg.TRACKS_DIR, stem + ".json")
    csv_path = os.path.join(cfg.TRACKS_DIR, stem + ".csv")
    overlay_path = os.path.join(cfg.OVERLAYS_DIR, stem + ".mp4")

    if os.path.exists(json_path) and not args.redo:
        print(f"  [skip] {rel_path} (already done)")
        return
    if not os.path.exists(abs_path):
        print(f"  [MISSING] {rel_path}")
        return

    print(f"\n=== {group}  |  {rel_path} ===")

    # Headless re-track of a clip whose saved tip box was fine but drifted:
    # reuse the stored box + the (now adaptive) tracker, no clicks needed.
    saved_box = None
    if args.from_saved_box:
        if not os.path.exists(json_path):
            print("  [skip] --from-saved-box but no existing result to reuse")
            return
        with open(json_path) as f:
            prev = json.load(f)
        saved_box = prev.get("tip_bbox")
        if not saved_box:
            print("  [skip] saved result has no tip_bbox (was it manual?)")
            return

    frame = first_frame(abs_path)
    win = None if (args.from_saved_box and not args.manual) else open_window("calibrate + select")

    # --- calibration ---
    override = cfg.GRID_OVERRIDE.get(rel_path)
    if override:
        px_per_sq, calib = float(override), dict(method="config_override",
                                                 confidence=100.0)
        print(f"  grid override from config: {px_per_sq} px/square")
    else:
        px_per_sq, calib = calibrate(win, frame, args.calib)
    if not px_per_sq:
        print("  [WARN] no calibration; using fallback 24 px/square. "
              "Set deflection_config.GRID_OVERRIDE for this clip if needed.")
        px_per_sq = 24.0
    elif calib.get("method") == "auto" and calib.get("confidence", 0) < 30:
        print(f"  [WARN] low auto-grid confidence ({calib['confidence']:.0f}); "
              f"got {px_per_sq:.1f} px/square. Consider --calib ruler.")
    px_per_mm = px_per_sq / cfg.GRID_MM_PER_SQUARE
    print(f"  scale: {px_per_sq:.2f} px/square -> {px_per_mm:.2f} px/mm "
          f"({calib.get('method')})")

    # --- track ---
    def prog(i, n):
        print(f"    tracking {i}/{n}", end="\r")

    tip_box = None
    keyframes = None
    if args.manual:
        if win is not None:
            cv2.destroyWindow(win)
        keyframes = manual_annotate(abs_path, args.manual_step)
        if len(keyframes) < 2:
            print("  [skip] manual mode needs >= 2 clicks")
            return
        track = wd.track_from_keyframes(abs_path, keyframes)
        print(f"    manual: {len(keyframes)} keyframes -> {track['nframes']} frames")
    else:
        if saved_box is not None:
            tip_box = tuple(saved_box)
            print(f"    reusing saved tip box {tip_box}")
        else:
            tip_box = select_roi(win, frame, "TIP: draw a SMALL TIGHT box on the tip, ENTER")
        if win is not None:
            cv2.destroyWindow(win)
        if tip_box[2] == 0 or tip_box[3] == 0:
            print("  [skip] no tip ROI selected")
            return
        if args.tracker == "csrt":
            track = wd.track_tip_csrt(abs_path, tip_box, progress=prog)
        elif args.tracker == "mil":
            track = wd.track_tip_mil(abs_path, tip_box, progress=prog)
        else:
            track = wd.track_tip_ncc(abs_path, tip_box,
                                     search_radius=args.search_radius,
                                     template_update=args.template_update,
                                     progress=prog)
        print(f"    tracked {track['nframes']} frames; "
              f"median match score {np.median(track['score']):.2f}      ")

    # --- metrics ---
    m = wd.compute_metrics(track["x"], track["y"], px_per_mm,
                           rest_frames=cfg.REST_FRAMES)
    print(f"    max deflection {m['max_deflection_mm']:.1f} mm | "
          f"peak-to-peak {m['peak_to_peak_mm']:.1f} mm")

    # --- save ---
    os.makedirs(cfg.TRACKS_DIR, exist_ok=True)
    os.makedirs(cfg.OVERLAYS_DIR, exist_ok=True)
    fps = track["fps"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x_px", "y_px", "score",
                    "disp_mm", "dx_mm", "dy_mm"])
        for i in range(track["nframes"]):
            w.writerow([int(track["frame"][i]), track["frame"][i] / fps,
                        f"{track['x'][i]:.3f}", f"{track['y'][i]:.3f}",
                        f"{track['score'][i]:.4f}", f"{m['disp_mm'][i]:.4f}",
                        f"{m['dx_mm'][i]:.4f}", f"{m['dy_mm'][i]:.4f}"])

    meta = dict(
        group=group, rel_path=rel_path, fps=fps,
        width=track["width"], height=track["height"], nframes=track["nframes"],
        tip_bbox=([int(v) for v in tip_box] if tip_box else None),
        calibration=calib,
        grid_mm_per_square=cfg.GRID_MM_PER_SQUARE,
        px_per_square=float(px_per_sq), px_per_mm=float(px_per_mm),
        grid_confidence=float(calib.get("confidence", 0.0)),
        mode=("manual" if args.manual else args.tracker),
        search_radius=args.search_radius,
        median_score=float(np.median(track["score"])),
        rest_frames=cfg.REST_FRAMES,
        rest_x=m["rest_x"], rest_y=m["rest_y"],
        max_deflection_mm=m["max_deflection_mm"],
        peak_to_peak_mm=m["peak_to_peak_mm"],
        max_deflection_frame=m["max_deflection_frame"],
    )
    if keyframes is not None:
        meta["keyframes"] = {int(k): list(v) for k, v in keyframes.items()}
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    if not args.no_overlay:
        print("    writing overlay...", end="\r")
        wd.write_overlay(abs_path, overlay_path, track, px_per_mm,
                         (m["rest_x"], m["rest_y"]), every=args.overlay_every)
        print(f"    overlay -> {os.path.relpath(overlay_path, cfg.ROOT)}   ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="process only this relative path")
    ap.add_argument("--redo", action="store_true", help="redo finished videos")
    ap.add_argument("--calib", choices=["ruler", "auto"], default="ruler")
    ap.add_argument("--manual", action="store_true",
                    help="hand-annotate the tip every N frames (bulletproof)")
    ap.add_argument("--from-saved-box", action="store_true",
                    help="reuse the tip box from the existing result and re-track "
                         "headlessly (no clicks); good after improving tracker settings")
    ap.add_argument("--manual-step", type=int, default=15,
                    help="frames between manual clicks (smaller = more accurate)")
    ap.add_argument("--tracker", choices=["csrt", "ncc", "mil"], default="csrt")
    ap.add_argument("--search-radius", type=int, default=180)
    ap.add_argument("--template-update", type=float, default=0.12,
                    help="0..1 adaptive template blend (0 = fixed template)")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--overlay-every", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(cfg.TRACKS_DIR, exist_ok=True)
    os.makedirs(cfg.OVERLAYS_DIR, exist_ok=True)

    todo = [(g, a, r) for (g, a, r) in cfg.all_videos()
            if (args.video is None or r == args.video)]
    if not todo:
        print("no matching videos (check --video path)")
        return
    print(f"{len(todo)} video(s) to consider")
    for group, abs_path, rel_path in todo:
        try:
            process(group, abs_path, rel_path, args)
        except Exception as e:
            print(f"  [ERROR] {rel_path}: {e}")
    print("\ndone. Review overlays in deflection_analysis/overlays/, "
          "then run the analysis notebook.")


if __name__ == "__main__":
    main()
