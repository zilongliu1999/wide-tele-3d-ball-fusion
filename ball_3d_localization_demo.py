# -*- coding: utf-8 -*-
"""
ball_3d_localization_demo.py
============================
Minimal self-contained demo: Wide-Tele Dual-Camera 3D Ball Localization

Three core modules demonstrated here:
  1. Subpixel Ball Detection   — HSV likelihood map + radial profile sampling
                                 + IRLS circle fitting
  2. Wide-Tele Image Registration — multi-scale template matching with
                                    PSR scoring + ECC refinement
  3. 3D Fusion                 — angular-radius distance (tele) × wide ray

All functions run on synthetic/toy inputs so you can execute this file
with only opencv-python and numpy installed:

    pip install opencv-python numpy
    python ball_3d_localization_demo.py

Author: <your name>
"""

import math
import cv2
import numpy as np


# ============================================================
# MODULE 1 — Subpixel Ball Detection
# ============================================================

def compute_red_likelihood(img_bgr: np.ndarray,
                           hue_width: float = 20.0,
                           v_gamma: float = 0.5) -> np.ndarray:
    """
    Per-pixel 'redness' likelihood in [0, 1].

    Combines hue closeness to red (0° / 180°) with saturation and a
    value-gamma term to down-weight dark pixels.

    Parameters
    ----------
    img_bgr   : uint8 BGR image
    hue_width : half-width of the red hue band (OpenCV H ∈ [0, 180])
    v_gamma   : exponent applied to value channel (0 = ignore brightness)

    Returns
    -------
    L : float32 array, same H×W as input
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32)
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0

    # Circular distance to red (handles wrap-around at 0° / 180°)
    d = np.minimum(np.abs(h), np.abs(h - 180.0))
    closeness = np.clip(1.0 - d / float(hue_width), 0.0, 1.0)

    L = closeness * s * np.power(v, float(v_gamma)) if v_gamma > 0 else closeness * s
    return L


def _bilinear_sample(img_f32: np.ndarray,
                     xs: np.ndarray,
                     ys: np.ndarray) -> np.ndarray:
    """Sub-pixel bilinear sampling on a float32 image."""
    H, W = img_f32.shape
    xs = np.clip(xs, 0.0, W - 2 - 1e-6)
    ys = np.clip(ys, 0.0, H - 2 - 1e-6)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)
    return (img_f32[y0, x0] * wa + img_f32[y1, x0] * wb +
            img_f32[y0, x1] * wc + img_f32[y1, x1] * wd)


def _parabola_subpixel(y_m1: float, y_0: float, y_p1: float) -> float:
    """Sub-pixel peak refinement via 3-point parabola fit."""
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_m1 - y_p1) / denom, -1.0, 1.0))


def robust_circle_fit(boundary_pts: list,
                      cx0: float, cy0: float, r0: float,
                      max_iter: int = 40,
                      huber_k: float = 1.345) -> tuple:
    """
    IRLS (Iteratively Re-weighted Least Squares) circle fit with Huber weights.

    Fits (cx, cy, r) to a set of 2-D boundary points by minimising a
    robust residual  ||p - c|| - r  with Huber M-estimator weights.

    Parameters
    ----------
    boundary_pts : list of (x, y) tuples
    cx0, cy0, r0 : initial estimate (e.g. from cv2.minEnclosingCircle)
    max_iter     : IRLS iterations
    huber_k      : Huber threshold in units of MAD-estimated sigma

    Returns
    -------
    cx, cy, r  : fitted circle parameters
    residuals  : per-point signed residuals (d - r)
    std        : [std_cx, std_cy, std_r] from covariance propagation
    """
    pts = np.asarray(boundary_pts, dtype=np.float64)
    c = np.array([cx0, cy0], dtype=np.float64)
    r = float(r0)

    for _ in range(max_iter):
        dx, dy = c[0] - pts[:, 0], c[1] - pts[:, 1]
        d = np.hypot(dx, dy) + 1e-12
        res = d - r

        # Robust scale via MAD
        sigma = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
        t = np.abs(res) / sigma
        w = np.where(t > huber_k, huber_k / t, np.ones_like(t))

        # Weighted Gauss-Newton step
        sqw = np.sqrt(w)
        J = np.column_stack([dx / d, dy / d, -np.ones(len(d))])
        JW = J * sqw[:, None]
        delta = np.linalg.solve(JW.T @ JW + 1e-6 * np.eye(3),
                                JW.T @ (-res * sqw))
        c += delta[:2]
        r = max(float(r + delta[2]), 1.0)
        if np.linalg.norm(delta) < 1e-6:
            break

    # Covariance for uncertainty estimate
    dx, dy = c[0] - pts[:, 0], c[1] - pts[:, 1]
    d = np.hypot(dx, dy) + 1e-12
    res = d - r
    sigma = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
    t = np.abs(res) / sigma
    w = np.where(t > huber_k, huber_k / t, np.ones_like(t))
    J = np.column_stack([dx / d, dy / d, -np.ones(len(d))])
    dof = max(len(res) - 3, 1)
    sigma2 = float(np.sum(w * res ** 2) / dof)
    Cov = np.linalg.pinv((J.T * w) @ J) * sigma2
    std = np.sqrt(np.clip(np.diag(Cov), 0, np.inf))

    return float(c[0]), float(c[1]), float(r), res, std


def detect_ball_subpixel(img_bgr: np.ndarray,
                         theta_bins: int = 360,
                         band_px: float = 20.0,
                         step_px: float = 0.5,
                         alpha: float = 0.25) -> dict | None:
    """
    Full subpixel ball detection pipeline on one image.

    Steps:
      1. HSV segmentation → coarse bounding circle
      2. Red-likelihood map L
      3. Radial profile sampling at `theta_bins` angles
      4. Sub-pixel edge via gradient peak + parabola refinement + cross threshold
      5. IRLS robust circle fit on boundary points

    Parameters
    ----------
    img_bgr    : BGR input image
    theta_bins : angular resolution for radial sampling
    band_px    : search band half-width around coarse radius
    step_px    : radial sampling step (pixels)
    alpha      : threshold blend ratio for cross-threshold refinement

    Returns
    -------
    dict with keys: cx, cy, r, std_cx, std_cy, std_r, n_pts
    or None if no valid ball found.
    """
    # --- Coarse detection ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 40), (12, 255, 255))
    m2 = cv2.inRange(hsv, (160, 70, 40), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best_cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best_cnt) < 500:
        return None
    (cx0, cy0), r0 = cv2.minEnclosingCircle(best_cnt)

    # --- Likelihood map ---
    L = compute_red_likelihood(img_bgr).astype(np.float64)

    # --- Radial profile sampling ---
    rs = np.arange(max(1.0, r0 - band_px), r0 + band_px + 1e-9, step_px)
    boundary_pts = []

    for bi in range(theta_bins):
        th = 2.0 * math.pi * bi / theta_bins
        cos_t, sin_t = math.cos(th), math.sin(th)
        xs = cx0 + rs * cos_t
        ys = cy0 + rs * sin_t
        prof = _bilinear_sample(L, xs, ys)

        # Smooth with a small Gaussian
        prof_f = prof.astype(np.float32).reshape(-1, 1)
        prof = cv2.GaussianBlur(prof_f, (1, 7), 1.2).reshape(-1).astype(np.float64)

        k_fringe = max(4, int(0.1 * len(prof)))
        Lin  = float(np.median(prof[:k_fringe]))
        Lout = float(np.median(prof[-k_fringe:]))
        if Lin < 0.06 or Lout > 0.05 or (Lin - Lout) < 0.03:
            continue

        g = (prof[1:] - prof[:-1]) / step_px
        idx = int(np.argmin(g))
        if -g[idx] < 0.01:
            continue

        # Parabola sub-pixel
        delta = 0.0
        if 1 <= idx < len(g) - 1:
            delta = _parabola_subpixel(-g[idx-1], -g[idx], -g[idx+1])
        r_edge = rs[idx] + (0.5 + delta) * step_px

        # Cross-threshold refinement in a local window
        tval = Lout + alpha * (Lin - Lout)
        win = 12
        for i in range(max(0, idx - win), min(len(prof) - 2, idx + win) + 1):
            if prof[i] >= tval > prof[i + 1]:
                frac = (prof[i] - tval) / (prof[i] - prof[i + 1] + 1e-12)
                r_edge = rs[i] + frac * step_px
                break

        boundary_pts.append((cx0 + r_edge * cos_t, cy0 + r_edge * sin_t))

    if len(boundary_pts) < 60:
        return None

    # --- IRLS circle fit ---
    cx, cy, r, _, std = robust_circle_fit(boundary_pts, cx0, cy0, r0)
    return dict(cx=cx, cy=cy, r=r,
                std_cx=float(std[0]), std_cy=float(std[1]), std_r=float(std[2]),
                n_pts=len(boundary_pts))


# ============================================================
# MODULE 2 — Wide-Tele Image Registration
# ============================================================

def _grad_mag(gray_u8: np.ndarray) -> np.ndarray:
    """Gradient-magnitude feature map (normalised uint8)."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _compute_psr(resp: np.ndarray, peak_xy: tuple, exclude_r: int = 10) -> float:
    """
    Peak-to-Sidelobe Ratio — measures template matching sharpness.

    A high PSR (> 8) indicates a confident, unambiguous match.
    """
    peak = float(resp[peak_xy[1], peak_xy[0]])
    mask = np.ones(resp.shape, bool)
    x, y = peak_xy
    mask[max(0, y-exclude_r):y+exclude_r+1,
         max(0, x-exclude_r):x+exclude_r+1] = False
    side = resp[mask]
    if side.size < 30:
        return float("nan")
    mu, sigma = float(side.mean()), float(side.std() + 1e-12)
    return (peak - mu) / sigma


def register_tele_to_wide(wide_gray: np.ndarray,
                          tele_gray: np.ndarray,
                          tele_ball_xy: tuple,
                          wide_ball_xy: tuple,
                          s_prior: float = 0.25,
                          n_scales: int = 25,
                          s_range: float = 0.40,
                          search_pad: int = 100) -> dict | None:
    """
    Multi-scale template matching with PSR scoring and ECC refinement.

    Algorithm:
      1. Build a log-spaced scale grid around `s_prior`
      2. At each scale, resize tele and run cv2.matchTemplate around the
         predicted wide-image position of the tele ball centre
      3. Score = NCC + w_psr * PSR + log-Gaussian prior on scale
      4. ECC (Enhanced Correlation Coefficient) sub-pixel refinement on the
         best scale to obtain a full 2×3 affine warp

    Parameters
    ----------
    wide_gray    : grayscale wide image
    tele_gray    : grayscale tele image
    tele_ball_xy : (u, v) of ball centre in tele image
    wide_ball_xy : (u, v) of ball centre in wide image
    s_prior      : expected tele/wide scale ratio
    n_scales     : number of scales to search
    s_range      : relative half-range around s_prior
    search_pad   : search window extension (px) around predicted position

    Returns
    -------
    dict: s, ncc, psr, x, y, tw, th, ecc_cc, warp_2x3  — or None on failure
    """
    wide_feat = _grad_mag(wide_gray)
    tele_feat = _grad_mag(tele_gray)

    cx_t, cy_t = float(tele_ball_xy[0]), float(tele_ball_xy[1])
    cx_w, cy_w = float(wide_ball_xy[0]), float(wide_ball_xy[1])

    s_min = max(0.05, s_prior * (1 - s_range))
    s_max = min(0.95, s_prior * (1 + s_range))
    scales = np.linspace(s_min, s_max, n_scales).tolist()

    best = None
    best_resp = None

    for s in scales:
        th = int(round(tele_feat.shape[0] * s))
        tw = int(round(tele_feat.shape[1] * s))
        if th < 60 or tw < 60:
            continue

        tele_s = cv2.resize(tele_feat, (tw, th), interpolation=cv2.INTER_AREA)

        # Predicted top-left corner in wide image
        x0 = int(round(cx_w - s * cx_t)) - search_pad
        y0 = int(round(cy_w - s * cy_t)) - search_pad
        x1, y1 = x0 + tw + 2 * search_pad, y0 + th + 2 * search_pad
        if x0 < 0 or y0 < 0 or x1 > wide_feat.shape[1] or y1 > wide_feat.shape[0]:
            continue

        roi = wide_feat[y0:y1, x0:x1]
        resp = cv2.matchTemplate(roi, tele_s, cv2.TM_CCOEFF_NORMED)
        _, ncc, _, loc = cv2.minMaxLoc(resp)
        psr = _compute_psr(resp, loc)

        # Log-Gaussian scale prior
        sigma_s = max(1e-9, s_prior * 0.12)
        log_prior = -0.5 * ((s - s_prior) / sigma_s) ** 2
        score = ncc + 0.02 * (psr if np.isfinite(psr) else 0.0) + 0.10 * log_prior

        cand = dict(s=float(s), ncc=float(ncc),
                    psr=float(psr) if np.isfinite(psr) else float("nan"),
                    score=float(score),
                    x=x0 + int(loc[0]), y=y0 + int(loc[1]),
                    tw=tw, th=th)
        if best is None or score > best["score"]:
            best = cand
            best_resp = resp

    if best is None:
        return None

    # ECC sub-pixel refinement
    x, y, tw, th = best["x"], best["y"], best["tw"], best["th"]
    wide_patch = wide_gray[y:y+th, x:x+tw]
    tele_scaled = cv2.resize(tele_gray, (tw, th), interpolation=cv2.INTER_AREA)
    if wide_patch.shape != tele_scaled.shape:
        return best

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        ecc_cc, warp = cv2.findTransformECC(
            tele_scaled.astype(np.float32) / 255.0,
            wide_patch.astype(np.float32) / 255.0,
            warp, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
        best["ecc_cc"] = float(ecc_cc)
        best["warp_2x3"] = warp
    except cv2.error:
        best["ecc_cc"] = float("nan")
        best["warp_2x3"] = None

    return best


# ============================================================
# MODULE 3 — 3D Fusion (tele distance × wide ray)
# ============================================================

def undistort_to_unit_ray(u: float, v: float,
                          K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Map an image pixel (u, v) → normalised 3-D ray in camera frame.

    Uses cv2.undistortPoints for distortion correction, then normalises
    to unit length.
    """
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, D, P=None)
    x, y = float(und[0, 0, 0]), float(und[0, 0, 1])
    ray = np.array([x, y, 1.0])
    return ray / (np.linalg.norm(ray) + 1e-12)


def estimate_distance_from_angular_radius(
        center_uv: tuple,
        boundary_pts_uv: list,
        K: np.ndarray,
        D: np.ndarray,
        sphere_radius_m: float) -> tuple[float, float] | tuple[None, None]:
    """
    Compute camera-to-ball distance using the angular radius of the sphere.

        D = R / sin(alpha)

    where alpha is the median angle between the centre ray and boundary
    point rays (robust to outlier boundary points).

    Using the median across boundary points makes this estimate robust to
    HSV segmentation noise and partial occlusions.

    Parameters
    ----------
    center_uv       : (u, v) pixel of ball centre
    boundary_pts_uv : list of (u, v) subpixel boundary points
    K, D            : camera intrinsics and distortion coefficients
    sphere_radius_m : physical radius of the sphere (metres)

    Returns
    -------
    (distance_m, alpha_rad)  or  (None, None) if insufficient points
    """
    if len(boundary_pts_uv) < 20:
        return None, None

    r_c = undistort_to_unit_ray(center_uv[0], center_uv[1], K, D)
    alphas = []
    for (x, y) in boundary_pts_uv:
        r_b = undistort_to_unit_ray(float(x), float(y), K, D)
        dot = float(np.clip(np.dot(r_c, r_b), -1.0, 1.0))
        # atan2 form avoids acos precision loss at small angles
        sin_a = float(np.linalg.norm(np.cross(r_c, r_b)))
        a = math.atan2(sin_a, dot)
        if a > 1e-8:
            alphas.append(a)

    if len(alphas) < 20:
        return None, None

    alpha = float(np.median(alphas))
    distance = float(sphere_radius_m / (math.sin(alpha) + 1e-12))
    return distance, alpha


def fuse_3d_position(tele_ball_uv: tuple,
                     tele_boundary_pts: list,
                     K_tele: np.ndarray,
                     D_tele: np.ndarray,
                     H_tele_to_wide: np.ndarray,
                     K_wide: np.ndarray,
                     D_wide: np.ndarray,
                     sphere_radius_m: float) -> dict:
    """
    Dual-branch 3D ball centre localisation.

    Fusion strategy (robust to unknown inter-camera baseline):
      - **Depth**     : estimated from tele angular radius — high accuracy
                        because the ball occupies more pixels on tele
      - **Direction** : derived from wide image ray — stable absolute
                        pointing reference

    Steps:
      1. Estimate depth D from tele angular radius
      2. Map tele ball centre → wide pixel via homography H_tele_to_wide
      3. Undistort wide pixel → unit ray r_w
      4. Output 3D position: P_wide = D * r_w

    Parameters
    ----------
    tele_ball_uv     : (u, v) subpixel ball centre in tele image
    tele_boundary_pts: subpixel boundary points in tele image
    K_tele, D_tele   : tele camera intrinsics / distortion
    H_tele_to_wide   : 3×3 homography mapping tele → wide pixels
    K_wide, D_wide   : wide camera intrinsics / distortion
    sphere_radius_m  : physical sphere radius (metres)

    Returns
    -------
    dict with keys:
        P_wide_m   : (3,) array — ball centre in wide camera frame (metres)
        D_m        : scalar distance estimate from tele
        alpha_rad  : angular radius
        wide_uv    : projected wide pixel (for sanity check)
        success    : bool
    """
    # Step 1 — depth from tele
    D_m, alpha = estimate_distance_from_angular_radius(
        tele_ball_uv, tele_boundary_pts, K_tele, D_tele, sphere_radius_m)
    if D_m is None:
        return dict(success=False)

    # Step 2 — map tele centre to wide pixel via homography
    p = H_tele_to_wide @ np.array([tele_ball_uv[0], tele_ball_uv[1], 1.0])
    if abs(p[2]) < 1e-12:
        return dict(success=False)
    u_w, v_w = float(p[0] / p[2]), float(p[1] / p[2])

    # Step 3 — wide unit ray
    r_w = undistort_to_unit_ray(u_w, v_w, K_wide, D_wide)

    # Step 4 — 3D position
    P_wide = D_m * r_w
    X, Y, Z = P_wide
    u_proj = float(K_wide[0, 0] * X / Z + K_wide[0, 2])
    v_proj = float(K_wide[1, 1] * Y / Z + K_wide[1, 2])

    return dict(
        success=True,
        P_wide_m=P_wide,
        D_m=float(D_m),
        alpha_rad=float(alpha),
        wide_uv=(u_w, v_w),
        projected_uv=(u_proj, v_proj),
    )


# ============================================================
# Self-contained demo / smoke test
# ============================================================

def _make_synthetic_ball_image(h: int = 480, w: int = 640,
                               cx: float = 320, cy: float = 240,
                               r: float = 60,
                               noise_std: float = 6.0) -> np.ndarray:
    """Create a synthetic red ball on a grey background for testing."""
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)),
               (30, 30, 200), -1, cv2.LINE_AA)
    # Shading — darker at edges to mimic a sphere
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    shade = np.clip(1.0 - (dist / (r + 1e-9)) ** 2, 0.0, 1.0)
    overlay = img.copy().astype(np.float32)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 55 * shade, 0, 255)
    img = np.clip(overlay + np.random.normal(0, noise_std, img.shape),
                  0, 255).astype(np.uint8)
    return img


def demo():
    print("=" * 60)
    print("  Wide-Tele 3D Ball Localisation — Demo")
    print("=" * 60)
    np.random.seed(42)

    # ---- Module 1: subpixel detection ----
    print("\n[Module 1] Subpixel ball detection on synthetic image …")
    GT_CX, GT_CY, GT_R = 318.7, 241.3, 58.5
    img = _make_synthetic_ball_image(cx=GT_CX, cy=GT_CY, r=GT_R)
    result = detect_ball_subpixel(img, theta_bins=360, band_px=20.0, step_px=0.5)
    if result:
        print(f"  Ground truth : cx={GT_CX:.1f}  cy={GT_CY:.1f}  r={GT_R:.1f}")
        print(f"  Detected     : cx={result['cx']:.2f} ± {result['std_cx']:.2f}"
              f"  cy={result['cy']:.2f} ± {result['std_cy']:.2f}"
              f"  r={result['r']:.2f} ± {result['std_r']:.2f}"
              f"  (n_pts={result['n_pts']})")
        err = math.hypot(result['cx'] - GT_CX, result['cy'] - GT_CY)
        print(f"  Centre error : {err:.3f} px")
    else:
        print("  [WARN] Detection failed on synthetic image.")

    # ---- Module 2: registration ----
    print("\n[Module 2] Wide-tele registration on synthetic pair …")
    SCALE = 0.25
    wide_img = _make_synthetic_ball_image(h=600, w=800, cx=400, cy=300, r=15)
    tele_img = _make_synthetic_ball_image(h=600, w=800, cx=320, cy=240, r=58)
    wide_gray = cv2.cvtColor(wide_img, cv2.COLOR_BGR2GRAY)
    tele_gray = cv2.cvtColor(tele_img, cv2.COLOR_BGR2GRAY)

    reg = register_tele_to_wide(
        wide_gray, tele_gray,
        tele_ball_xy=(320, 240),
        wide_ball_xy=(400, 300),
        s_prior=SCALE, n_scales=15, search_pad=60)
    if reg:
        print(f"  Best scale   : {reg['s']:.3f}  (target ≈ {SCALE:.3f})")
        print(f"  NCC          : {reg['ncc']:.4f}")
        print(f"  PSR          : {reg['psr']:.2f}" if np.isfinite(reg.get('psr', float('nan')))
              else "  PSR          : n/a")
        if reg.get("ecc_cc") and np.isfinite(reg["ecc_cc"]):
            print(f"  ECC ρ        : {reg['ecc_cc']:.4f}")
    else:
        print("  [WARN] Registration failed on synthetic pair.")

    # ---- Module 3: 3D fusion ----
    print("\n[Module 3] 3D fusion with known camera parameters …")
    f = 12000.0
    K_tele = np.array([[f, 0, 960], [0, f, 720], [0, 0, 1]], dtype=np.float64)
    K_wide = np.array([[1200, 0, 960], [0, 1200, 720], [0, 0, 1]], dtype=np.float64)
    D_zero = np.zeros((1, 5), dtype=np.float64)

    # Tele ball at 3 m directly ahead
    Z_GT = 3.0
    SPHERE_R = 0.10
    cx_t = float(K_tele[0, 2])   # ball on tele optical axis
    cy_t = float(K_tele[1, 2])
    alpha_gt = math.asin(SPHERE_R / Z_GT)
    r_tele_px = math.tan(alpha_gt) * float(K_tele[0, 0])

    # Synthetic boundary points
    boundary = [(cx_t + r_tele_px * math.cos(2 * math.pi * i / 360),
                 cy_t + r_tele_px * math.sin(2 * math.pi * i / 360))
                for i in range(360)]

    # Identity homography (tele == wide for this test)
    H = np.eye(3, dtype=np.float64)

    out = fuse_3d_position(
        tele_ball_uv=(cx_t, cy_t),
        tele_boundary_pts=boundary,
        K_tele=K_tele, D_tele=D_zero,
        H_tele_to_wide=H,
        K_wide=K_wide, D_wide=D_zero,
        sphere_radius_m=SPHERE_R)

    if out["success"]:
        P = out["P_wide_m"]
        print(f"  Ground truth : Z = {Z_GT:.3f} m")
        print(f"  Estimated    : X={P[0]:.4f}  Y={P[1]:.4f}  Z={P[2]:.4f} m")
        print(f"  Depth error  : {abs(P[2] - Z_GT)*100:.2f} cm")
        print(f"  Alpha        : {math.degrees(out['alpha_rad']):.4f}°")
    else:
        print("  [WARN] 3D fusion failed.")

    print("\n[Done] All three modules completed successfully.")


if __name__ == "__main__":
    demo()
