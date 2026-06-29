"""
Core routines for the pneumatic chicken-wing tip-deflection benchmark.

Four jobs:
  1. detect_grid_px  -- find graph-paper square size in pixels (FFT) -> px/mm
  2. track_tip       -- follow a user-selected tip ROI through a video
                        (normalized cross-correlation by default, MIL optional)
  3. compute_metrics -- max-deflection-from-rest and peak-to-peak travel (mm)
  4. segment_wing    -- isolate the wing tissue from the graph-paper background
                        and measure its size (area, sqrt-area, caliper length)

Nothing here is interactive; track_tips.py / wing_size.py drive the clicking & I/O.
"""
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. Grid calibration
# ---------------------------------------------------------------------------
def measure_grid(gray, lo=38, hi=58):
    """Robust px-per-square by measuring spacing between dark grid lines.

    Averages many clean strips across the frame, collects all adjacent-line
    spacings, and returns the MEDIAN of those falling in [lo, hi] px (which
    rejects wing/clutter/partial detections). For this footage's clean, uniform
    graph paper this is far more reliable than FFT/autocorr. Returns
    (px_per_square, n_samples); px_per_square is None if too few clean samples.

    lo/hi bracket the expected square size in px -- widen them if the camera
    distance changes a lot.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return None, 0
    g = gray.astype(np.float32)
    H, W = g.shape
    spac = []
    for y in range(60, H - 40, 60):                      # horizontal strips
        s = g[y:y + 20, int(W * 0.45):W - 20].mean(axis=0)
        det = s - cv2.GaussianBlur(s.reshape(1, -1), (0, 0), 20).ravel()
        mins, _ = find_peaks(-det, distance=lo - 8, prominence=1)
        spac += list(np.diff(mins))
    for x in range(int(W * 0.5), W - 20, 80):            # vertical strips
        s = g[40:H - 40, x:x + 20].mean(axis=1)
        det = s - cv2.GaussianBlur(s.reshape(1, -1), (0, 0), 20).ravel()
        mins, _ = find_peaks(-det, distance=lo - 8, prominence=1)
        spac += list(np.diff(mins))
    spac = np.array(spac)
    clean = spac[(spac >= lo) & (spac <= hi)]
    if len(clean) < 5:
        return None, int(len(clean))
    return float(np.median(clean)), int(len(clean))


def ruler_px_per_square(p1, p2, n_squares):
    """px per grid square from two clicked points spanning `n_squares` squares.

    The most reliable calibration: immune to faint/harmonic grid lines.
    """
    d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return d / max(1e-6, n_squares)


def _autocorr_period(profile, min_px=6, max_px=70):
    """Fundamental spatial period (px) of a 1-D profile via autocorrelation.

    The FIRST autocorrelation peak is the fundamental grid spacing, so this is
    immune to the 2x/3x harmonics that an FFT-peak picker latches onto on small
    or low-contrast patches. Returns (period_px, peak_height in 0..1).
    """
    p = profile.astype(np.float64)
    p = p - p.mean()
    n = len(p)
    if n < 2 * min_px:
        return None, 0.0
    F = np.fft.rfft(p, n=2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    hi = min(max_px, n - 2)
    best = (None, 0.0)
    for L in range(min_px, hi):
        if ac[L] > ac[L - 1] and ac[L] >= ac[L + 1] and ac[L] > 0.05:
            return float(L), float(ac[L])  # first peak == fundamental
    return best


def detect_grid_px(gray_roi, min_px=6, max_px=70):
    """Estimate graph-paper square size (px) from a clean background ROI.

    Uses autocorrelation of the high-pass row/column-mean profiles and takes the
    FUNDAMENTAL period (first peak) on each axis. Averages the two axes when they
    agree to within 15%, else trusts the stronger peak. Confidence (0..100)
    folds in the autocorrelation peak height AND how many grid periods the patch
    spans -- a 2-square patch can never be trusted.

    NOTE: a 2-click ruler calibration (ruler_px_per_square) is more reliable;
    use this only for a quick auto estimate, and check the confidence.
    """
    roi = gray_roi.astype(np.float32)
    hp = roi - cv2.GaussianBlur(roi, (0, 0), sigmaX=6)
    px_x, ax = _autocorr_period(hp.mean(axis=0), min_px, max_px)  # vertical lines
    px_y, ay = _autocorr_period(hp.mean(axis=1), min_px, max_px)  # horizontal lines

    cands = [(p, a) for p, a in ((px_x, ax), (px_y, ay)) if p]
    if not cands:
        return dict(px_per_square=None, px_x=px_x, px_y=px_y,
                    amp_x=ax, amp_y=ay, confidence=0.0, n_periods=0.0)

    if len(cands) == 2 and abs(cands[0][0] - cands[1][0]) / max(
            cands[0][0], cands[1][0]) < 0.15:
        (pa, aa), (pb, ab) = cands
        px = (pa * aa + pb * ab) / (aa + ab)
        peak = max(aa, ab)
    else:
        px, peak = max(cands, key=lambda c: c[1])

    span = min(roi.shape)
    n_periods = span / px if px else 0.0
    # confidence: peak height * saturating bonus for spanning many squares
    conf = 100.0 * peak * min(1.0, n_periods / 6.0)
    return dict(px_per_square=float(px), px_x=px_x, px_y=px_y,
                amp_x=ax, amp_y=ay, confidence=float(conf),
                n_periods=float(n_periods))


# ---------------------------------------------------------------------------
# 2. Tip tracking
# ---------------------------------------------------------------------------
def _subpixel_peak(score, iy, ix):
    """Parabolic sub-pixel refinement of a correlation-map peak at (iy, ix)."""
    dx = dy = 0.0
    h, w = score.shape
    if 0 < ix < w - 1:
        l, c, r = score[iy, ix - 1], score[iy, ix], score[iy, ix + 1]
        denom = (l - 2 * c + r)
        if denom != 0:
            dx = 0.5 * (l - r) / denom
    if 0 < iy < h - 1:
        u, c, d = score[iy - 1, ix], score[iy, ix], score[iy + 1, ix]
        denom = (u - 2 * c + d)
        if denom != 0:
            dy = 0.5 * (u - d) / denom
    return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))


def track_tip_ncc(video_path, init_bbox, search_radius=180,
                  template_update=0.12, score_floor=0.30, progress=None):
    """Track a tip ROI via normalized cross-correlation (TM_CCOEFF_NORMED).

    init_bbox    : (x, y, w, h) ROI in the FIRST frame, centered on the tip.
    search_radius: how many px around the last position to search each frame.
    template_update: 0..1 blend of current patch into the template (0 = fixed).
    score_floor  : if best match < this, hold previous position, flag low conf.

    Returns dict with arrays: frame, x, y (tip center, subpixel px), score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ok, frame = cap.read()
    if not ok:
        raise IOError(f"cannot read first frame of {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    x0, y0, w, h = [int(round(v)) for v in init_bbox]
    template = gray[y0:y0 + h, x0:x0 + w].astype(np.float32)
    tmpl_f = template.copy()
    cx, cy = x0 + w / 2.0, y0 + h / 2.0   # tip center

    xs, ys, ss, fr = [cx], [cy], [1.0], [0]
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # search window around last center
        sx0 = int(max(0, cx - w / 2 - search_radius))
        sy0 = int(max(0, cy - h / 2 - search_radius))
        sx1 = int(min(W, cx + w / 2 + search_radius))
        sy1 = int(min(H, cy + h / 2 + search_radius))
        win = gray[sy0:sy1, sx0:sx1]
        if win.shape[0] < h or win.shape[1] < w:
            xs.append(cx); ys.append(cy); ss.append(0.0); fr.append(idx); continue

        res = cv2.matchTemplate(win, tmpl_f, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        mx, my = maxloc
        ddx, ddy = _subpixel_peak(res, my, mx)
        new_x = sx0 + mx + ddx + w / 2.0
        new_y = sy0 + my + ddy + h / 2.0

        if maxval >= score_floor:
            cx, cy = new_x, new_y
            if template_update > 0:     # conservative adaptive template
                tx0 = int(round(cx - w / 2)); ty0 = int(round(cy - h / 2))
                patch = gray[ty0:ty0 + h, tx0:tx0 + w]
                if patch.shape == tmpl_f.shape:
                    tmpl_f = ((1 - template_update) * tmpl_f
                              + template_update * patch).astype(np.float32)
        # else: low confidence -> hold previous center

        xs.append(cx); ys.append(cy); ss.append(float(maxval)); fr.append(idx)
        if progress and idx % 50 == 0:
            progress(idx, nframes)
    cap.release()

    return dict(
        frame=np.array(fr), x=np.array(xs), y=np.array(ys),
        score=np.array(ss), fps=float(fps), width=W, height=H,
        nframes=len(fr), bbox=(x0, y0, w, h),
    )


def track_tip_csrt(video_path, init_bbox, progress=None):
    """Track the tip with OpenCV's CSRT tracker (needs opencv-contrib).

    CSRT handles rotation and appearance change far better than rigid template
    matching, so it holds the tip through the large, fast deflections that defeat
    the NCC tracker. Center of the reported box is the tip. score = 1 (tracked)
    or 0 (lost).
    """
    if not hasattr(cv2, "TrackerCSRT_create"):
        raise RuntimeError("CSRT unavailable; pip install opencv-contrib-python")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"cannot read first frame of {video_path}")
    H, W = frame.shape[:2]
    x0, y0, w, h = [int(round(v)) for v in init_bbox]
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, (x0, y0, w, h))
    xs, ys, ss, fr = [x0 + w / 2.0], [y0 + h / 2.0], [1.0], [0]
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        found, box = tracker.update(frame)
        if found:
            bx, by, bw, bh = box
            xs.append(bx + bw / 2.0); ys.append(by + bh / 2.0); ss.append(1.0)
        else:
            xs.append(xs[-1]); ys.append(ys[-1]); ss.append(0.0)
        fr.append(idx)
        if progress and idx % 50 == 0:
            progress(idx, nframes)
    cap.release()
    return dict(frame=np.array(fr), x=np.array(xs), y=np.array(ys),
                score=np.array(ss), fps=float(fps), width=W, height=H,
                nframes=len(fr), bbox=(x0, y0, w, h))


def track_tip_mil(video_path, init_bbox, progress=None):
    """Fallback tracker using OpenCV's built-in MIL (no contrib needed)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"cannot read first frame of {video_path}")
    H, W = frame.shape[:2]
    x0, y0, w, h = [int(round(v)) for v in init_bbox]
    tracker = cv2.TrackerMIL_create()
    tracker.init(frame, (x0, y0, w, h))
    xs, ys, ss, fr = [x0 + w / 2.0], [y0 + h / 2.0], [1.0], [0]
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        found, box = tracker.update(frame)
        if found:
            bx, by, bw, bh = box
            xs.append(bx + bw / 2.0); ys.append(by + bh / 2.0); ss.append(1.0)
        else:
            xs.append(xs[-1]); ys.append(ys[-1]); ss.append(0.0)
        fr.append(idx)
        if progress and idx % 50 == 0:
            progress(idx, nframes)
    cap.release()
    return dict(frame=np.array(fr), x=np.array(xs), y=np.array(ys),
                score=np.array(ss), fps=float(fps), width=W, height=H,
                nframes=len(fr), bbox=(x0, y0, w, h))


def track_from_keyframes(video_path, keyframes):
    """Build a per-frame tip track by interpolating manual click keyframes.

    keyframes : dict {frame_index: (x_px, y_px)}. Must include >= 2 frames.
    Positions between clicks are linearly interpolated; this is the bulletproof
    fallback for clips the automatic tracker can't hold. Accuracy on the
    deflection EXTREMES just needs a click at (or near) the rest and peak frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    kf = sorted(keyframes.items())
    if len(kf) < 2:
        raise ValueError("need at least 2 keyframes")
    if n <= 0:
        n = kf[-1][0] + 1
    fr = np.arange(n)
    kx = np.array([f for f, _ in kf])
    xs = np.interp(fr, kx, np.array([p[0] for _, p in kf]))
    ys = np.interp(fr, kx, np.array([p[1] for _, p in kf]))
    score = np.zeros(n)
    score[kx[kx < n]] = 1.0          # mark which frames were hand-clicked
    return dict(frame=fr, x=xs, y=ys, score=score, fps=float(fps),
                width=W, height=H, nframes=n, bbox=None,
                keyframes={int(f): [float(p[0]), float(p[1])] for f, p in kf})


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(x, y, px_per_mm, rest_frames=12):
    """Deflection metrics from a tip track (px) and calibration (px/mm).

    rest position = median of the first `rest_frames` samples.
      max_deflection_mm : largest distance of the tip from its rest position.
      peak_to_peak_mm   : full travel range along the dominant motion axis
                          (1st principal component of the (x,y) track).

    Returns dict of scalars + the per-frame mm signals for plotting.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(rest_frames, len(x))
    rx, ry = np.median(x[:n]), np.median(y[:n])

    dx, dy = x - rx, y - ry
    disp_mm = np.hypot(dx, dy) / px_per_mm
    max_defl = float(np.nanmax(disp_mm))
    max_idx = int(np.nanargmax(disp_mm))

    # principal motion axis via PCA on centered track
    pts = np.column_stack([x - x.mean(), y - y.mean()])
    if len(pts) >= 2 and np.any(pts):
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        proj = pts @ axis
        p2p_mm = float((proj.max() - proj.min()) / px_per_mm)
    else:
        p2p_mm = 0.0

    return dict(
        max_deflection_mm=max_defl,
        peak_to_peak_mm=p2p_mm,
        max_deflection_frame=max_idx,
        rest_x=float(rx), rest_y=float(ry),
        disp_mm=disp_mm,           # per-frame magnitude from rest
        dx_mm=dx / px_per_mm,
        dy_mm=dy / px_per_mm,
    )


# ---------------------------------------------------------------------------
# Overlay rendering (verification)
# ---------------------------------------------------------------------------
def write_overlay(video_path, out_path, track, px_per_mm, rest_xy,
                  every=1, max_frames=None):
    """Render a verification video: tracked tip + trail + live deflection (mm)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = track["width"], track["height"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps / max(1, every), (W, H))
    rx, ry = rest_xy
    xs, ys = track["x"], track["y"]
    trail = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(xs):
            break
        if max_frames and idx >= max_frames:
            break
        cx, cy = int(round(xs[idx])), int(round(ys[idx]))
        trail.append((cx, cy))
        if idx % every == 0:
            for j in range(1, len(trail)):
                cv2.line(frame, trail[j - 1], trail[j], (0, 200, 255), 1)
            cv2.circle(frame, (int(rx), int(ry)), 6, (255, 0, 0), 2)   # rest (blue)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)            # tip (red)
            cv2.line(frame, (int(rx), int(ry)), (cx, cy), (0, 255, 0), 1)
            d_mm = np.hypot(cx - rx, cy - ry) / px_per_mm
            cv2.putText(frame, f"deflection: {d_mm:5.1f} mm", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
            cv2.putText(frame, f"deflection: {d_mm:5.1f} mm", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            vw.write(frame)
        idx += 1
    cap.release()
    vw.release()


# ---------------------------------------------------------------------------
# 4. Wing size (segmentation)
# ---------------------------------------------------------------------------
def rest_frame_median(video_path, n=12):
    """Median of the first `n` frames -- a denoised 'wing at rest' still (BGR)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    frames = []
    for _ in range(max(1, n)):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise IOError(f"cannot read frames of {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _fill_holes(mask):
    """Fill interior holes of a binary mask (e.g. the balloon sitting on tissue)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def _caliper_length_px(mask):
    """Max caliper (Feret) length in px = farthest pair on the silhouette hull.

    Returns (length_px, (p1, p2)) or (0.0, None) if the mask is empty.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c).reshape(-1, 2)
    if len(hull) < 2:
        return 0.0, None
    D = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)
    a, b = np.unravel_index(int(np.argmax(D)), D.shape)
    return float(D[a, b]), (tuple(hull[a]), tuple(hull[b]))


def segment_wing(bgr, warmth_blur=21, min_area_frac=0.01):
    """Segment the chicken-wing tissue from the graph-paper background.

    Tissue is *warm* (reddish/yellow) while paper, shadow, the white balloon and
    the black mount are colour-neutral or dark. We score each pixel by warmth in
    LAB space -- (a*-128) + (b*-128) -- Otsu-threshold it, clean up, keep the
    largest connected component, and fill interior holes (so the balloon resting
    on the wing still counts as wing).

    Returns dict: mask (uint8 0/255, holes filled), area_px, caliper_px,
    caliper_pts, ok (False if nothing plausible was found).

    KNOWN FAILURE: wings wrapped in opaque white actuator rings get split into
    pieces -- only the largest survives, so size is underestimated. Use the
    wing_size.py --manual polygon fallback for those.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, A, B = cv2.split(lab)
    warm = np.clip((A.astype(np.int16) - 128) + (B.astype(np.int16) - 128),
                   0, 255).astype(np.uint8)
    warm = cv2.GaussianBlur(warm, (0, 0), 2)
    _, th = cv2.threshold(warm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

    n, lbls, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return dict(mask=th, area_px=0.0, caliper_px=0.0, caliper_pts=None,
                    ok=False)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lbls == big).astype(np.uint8) * 255
    mask = _fill_holes(mask)

    area_px = float(cv2.countNonZero(mask))
    ok = area_px >= min_area_frac * bgr.shape[0] * bgr.shape[1]
    cal_px, cal_pts = _caliper_length_px(mask)
    return dict(mask=mask, area_px=area_px, caliper_px=cal_px,
                caliper_pts=cal_pts, ok=ok)


def wing_size_metrics(seg, px_per_mm):
    """Convert a segment_wing() result to physical sizes (mm)."""
    area_mm2 = seg["area_px"] / (px_per_mm ** 2)
    return dict(
        area_mm2=area_mm2,
        sqrt_area_mm=float(np.sqrt(max(0.0, area_mm2))),
        caliper_length_mm=seg["caliper_px"] / px_per_mm,
    )


def draw_wing_overlay(bgr, seg, sizes):
    """Verification image: tissue contour (green) + caliper line (red) + sizes."""
    out = bgr.copy()
    cnts, _ = cv2.findContours(seg["mask"], cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    if seg.get("caliper_pts"):
        p1, p2 = seg["caliper_pts"]
        cv2.line(out, p1, p2, (0, 0, 255), 3)
    txt = (f"sqrt(area)={sizes['sqrt_area_mm']:.0f}mm  "
           f"len={sizes['caliper_length_mm']:.0f}mm  "
           f"area={sizes['area_mm2']:.0f}mm2")
    cv2.putText(out, txt, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, txt, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return out
