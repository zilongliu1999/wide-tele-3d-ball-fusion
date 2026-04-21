# -*- coding: utf-8 -*-
"""
generate_demo_visuals.py
========================
Runs the full wide-tele 3D localisation pipeline on real field images and
saves annotated figures for the GitHub README.

All algorithm functions are self-contained in this file — no external
pipeline script is required.

Usage
-----
    pip install opencv-python numpy
    python generate_demo_visuals.py

Place Img238.jpg (wide) and Img333.jpg (tele) in the same directory.

Output
------
All figures are written to ./demo_output/
  01_wide_detection.jpg      Wide image with all detected markers annotated
  02_tele_subpixel.jpg       Tele image with subpixel boundary + IRLS fit
  03_likelihood_map.jpg      Red-likelihood heatmap
  04_registration_blend.jpg  Tele warped onto wide via real H_total (TM+ECC)
  05_3d_result.jpg           3D positions annotated on wide image
"""

import os, math, time, csv
import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
WIDE_PATH = "./Img238.jpg"   # wide-angle input image
TELE_PATH = "./Img333.jpg"   # telephoto input image
OUT_DIR   = "./demo_output"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Camera parameters
# ══════════════════════════════════════════════════════════════════════════════

K_WIDE = np.array([[1.2207e4, 0.0, 2.7651e3],
                   [0.0, 1.2208e4, 1.8085e3],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
D_WIDE = np.array([[0.0199, 1.6548, 0.0, 0.0, 0.0]], dtype=np.float64)

K_TELE = np.array([[6.0411e4, 0.0, 1.9527e3],
                   [0.0, 6.0421e4, 1.3781e3],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
D_TELE = np.array([[0.1697, -1.3617e3, 0.0, 0.0, 0.0]], dtype=np.float64)

USE_DISTORTION    = False   # set True to enable distortion correction
SPHERE_DIAMETER_M = 0.20    # physical marker diameter in metres


# ══════════════════════════════════════════════════════════════════════════════
# Detection parameters
# ══════════════════════════════════════════════════════════════════════════════

RED_H1 = (0,   70, 40);  RED_H2 = (12,  255, 255)
RED_H3 = (160, 70, 40);  RED_H4 = (180, 255, 255)
MORPH_K = 7;  OPEN_IT = 1;  CLOSE_IT = 2
MIN_AREA = 800;  MAX_AREA = 3_000_000;  MIN_CIRCULARITY = 0.55
MIN_R0_PX = 18;  MIN_FILL_RATIO = 0.45
MIN_R_REL_TO_MAX = 0.35;  ENABLE_RELATIVE_FILTER = True
HUE_WIDTH = 20.0;  V_GAMMA = 0.5
THETA_BINS = 720;  BAND_PX = 26.0;  STEP_PX = 0.50
INNER_THR = 0.06;  OUTER_THR = 0.05;  CONTRAST_THR = 0.03
PROFILE_SMOOTH_SIGMA = 1.2;  MIN_DROP = 0.010
REFINE_WITH_CROSS = True;  CROSS_WIN = 14
ALPHA_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35]
R_EDGE_MAD_K = 3.5;  IRLS_ITERS = 40;  HUBER_K = 1.345
LAMBDA_R_AREA = 0.15
R_OUTLIER_REL_MAX = 1.35;  R_OUTLIER_REL_MIN = 0.65
MAX_TARGETS_PER_IMAGE = 20


# ══════════════════════════════════════════════════════════════════════════════
# Registration parameters
# ══════════════════════════════════════════════════════════════════════════════

SEARCH_PAD = 120;  S_PRIOR_DEFAULT = 0.24;  S_RANGE = 0.45;  N_SCALES = 33
S_PRIOR_SIGMA_FRAC = 0.12
W_PSR = 0.02;  W_PRIOR = 0.10
W_FINAL_MATCH = 0.15;  W_FINAL_PSR = 0.01;  W_FINAL_RMSE = 0.20
ECC_ITERS = 120;  ECC_EPS = 1e-6
ECC_MOTION = cv2.MOTION_EUCLIDEAN;  BORDER_MODE = cv2.BORDER_REPLICATE
BLEND_ALPHA = 0.45


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hsv_mask_red(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_H1, RED_H2)
    m2 = cv2.inRange(hsv, RED_H3, RED_H4)
    mask = cv2.bitwise_or(m1, m2)
    k = np.ones((MORPH_K, MORPH_K), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=OPEN_IT)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CLOSE_IT)
    return mask

def _circularity(contour):
    a = float(cv2.contourArea(contour))
    p = float(cv2.arcLength(contour, True)) + 1e-12
    return 4.0 * math.pi * a / (p * p)

def _red_likelihood(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32)
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0
    d = np.minimum(np.abs(h), np.abs(h - 180.0))
    closeness = np.clip(1.0 - d / HUE_WIDTH, 0.0, 1.0)
    return closeness * s * np.power(v, V_GAMMA)

def _bilinear(img_f, xs, ys):
    H, W = img_f.shape
    xs = np.clip(np.asarray(xs, np.float64), 0, W - 2 - 1e-6)
    ys = np.clip(np.asarray(ys, np.float64), 0, H - 2 - 1e-6)
    x0 = np.floor(xs).astype(np.int32);  y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1;  y1 = y0 + 1
    wa=(x1-xs)*(y1-ys); wb=(x1-xs)*(ys-y0); wc=(xs-x0)*(y1-ys); wd=(xs-x0)*(ys-y0)
    return img_f[y0,x0]*wa + img_f[y1,x0]*wb + img_f[y0,x1]*wc + img_f[y1,x1]*wd

def _smooth1d(prof, sigma):
    if sigma <= 0: return prof
    k = int(round(sigma * 6 + 1))
    if k < 3: return prof
    if k % 2 == 0: k += 1
    p = prof.astype(np.float32).reshape(-1, 1)
    return cv2.GaussianBlur(p, (1, k), sigma, borderType=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float64)

def _parabola_peak(ym1, y0, yp1):
    d = ym1 - 2.0*y0 + yp1
    return float(np.clip(0.5*(ym1-yp1)/d, -1.0, 1.0)) if abs(d) > 1e-12 else 0.0

def _filter_candidates(cand):
    kept = [(a,ci,cx,cy,r,c) for (a,ci,cx,cy,r,c) in cand
            if r >= MIN_R0_PX and a/(math.pi*r*r+1e-12) >= MIN_FILL_RATIO]
    if ENABLE_RELATIVE_FILTER and len(kept) >= 2:
        max_r = max(x[4] for x in kept)
        kept = [x for x in kept if x[4] >= max_r * MIN_R_REL_TO_MAX]
    return kept

def _irls_circle(pts, cx0, cy0, r0):
    pts = np.asarray(pts, np.float64)
    c = np.array([cx0, cy0], np.float64);  r = float(r0)
    for _ in range(IRLS_ITERS):
        dx = c[0]-pts[:,0];  dy = c[1]-pts[:,1]
        d = np.hypot(dx, dy) + 1e-12;  res = d - r
        sig = 1.4826*np.median(np.abs(res-np.median(res))) + 1e-12
        t = np.abs(res)/sig;  w = np.where(t>HUBER_K, HUBER_K/t, np.ones_like(t))
        sw = np.sqrt(w)
        J  = np.stack([dx/d, dy/d, -np.ones_like(d)], axis=1)
        JW = J * sw[:,None];  rw = (-res) * sw
        delta = np.linalg.solve(JW.T@JW + 1e-6*np.eye(3), JW.T@rw)
        c += delta[:2];  r = max(float(r+delta[2]), 1.0)
        if np.linalg.norm(delta) < 1e-6: break
    dx = c[0]-pts[:,0];  dy = c[1]-pts[:,1]
    d = np.hypot(dx,dy)+1e-12;  res = d-r
    sig = 1.4826*np.median(np.abs(res-np.median(res)))+1e-12
    t = np.abs(res)/sig;  w = np.where(t>HUBER_K, HUBER_K/t, np.ones_like(t))
    J = np.stack([dx/d, dy/d, -np.ones_like(d)], axis=1)
    dof = max(len(res)-3,1);  s2 = float(np.sum(w*res**2)/dof)
    std = np.sqrt(np.clip(np.diag(np.linalg.pinv((J.T*w)@J)*s2), 0, np.inf))
    return float(c[0]), float(c[1]), float(r), res, std

def _boundary_points(L, cx0, cy0, r0, alpha):
    band = max(BAND_PX, 0.20*r0)
    rs = np.arange(max(1.0, r0-band), r0+band+1e-9, STEP_PX)
    pts = [];  r_edges = []
    for bi in range(THETA_BINS):
        th = 2*math.pi*bi/THETA_BINS
        ct, st = math.cos(th), math.sin(th)
        prof = _smooth1d(_bilinear(L, cx0+rs*ct, cy0+rs*st), PROFILE_SMOOTH_SIGMA)
        n = len(prof)
        if n < 8: continue
        ki = max(6, int(0.10*n))
        Lin = float(np.median(prof[:ki]));  Lout = float(np.median(prof[-ki:]))
        if Lin < INNER_THR or Lout > OUTER_THR or (Lin-Lout) < CONTRAST_THR: continue
        g = (prof[1:]-prof[:-1])/STEP_PX
        idx = int(np.argmin(g))
        if -g[idx] < MIN_DROP: continue
        delta = 0.0
        if 1 <= idx < len(g)-1:
            delta = _parabola_peak(-g[idx-1], -g[idx], -g[idx+1])
        r_edge = rs[idx] + (0.5+delta)*STEP_PX
        if REFINE_WITH_CROSS:
            tval = Lout + alpha*(Lin-Lout)
            for i in range(max(0,idx-CROSS_WIN), min(len(prof)-2,idx+CROSS_WIN)+1):
                if prof[i] >= tval > prof[i+1]:
                    r_edge = rs[i] + (prof[i]-tval)/(prof[i]-prof[i+1]+1e-12)*STEP_PX
                    break
        r_edges.append(r_edge);  pts.append((cx0+r_edge*ct, cy0+r_edge*st))
    if len(pts) < 20:
        return pts, float(len(pts))/THETA_BINS
    r_edges = np.array(r_edges);  pts = np.array(pts)
    med = np.median(r_edges)
    keep = np.abs(r_edges-med) <= R_EDGE_MAD_K*1.4826*np.median(np.abs(r_edges-med)+1e-12)
    pts_k = pts[keep]
    return [tuple(p) for p in pts_k], float(len(pts_k))/THETA_BINS

def _pick_best_alpha(L, cx0, cy0, r0, area):
    r_area = math.sqrt(float(area)/math.pi)
    best = None
    for alpha in ALPHA_SWEEP:
        pts, cov = _boundary_points(L, cx0, cy0, r0, alpha)
        if len(pts) < 120 or cov < 0.35: continue
        cx, cy, r, res, std = _irls_circle(pts, cx0, cy0, r0)
        rms = float(np.sqrt(np.mean(res**2)));  mean = float(np.mean(res))
        if r > r_area*R_OUTLIER_REL_MAX or r < r_area*R_OUTLIER_REL_MIN: continue
        score = rms + 0.5*abs(mean) + LAMBDA_R_AREA*abs(r-r_area)
        if best is None or score < best["score"]:
            best = dict(alpha=alpha, pts=pts, coverage=cov,
                        cx=cx, cy=cy, r=r, std=std, rms=rms, score=score)
    return best

def detect_wide_targets(wide_bgr):
    mask = _hsv_mask_red(wide_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 400: continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 8: continue
        x, y, w, h = cv2.boundingRect(c)
        targets.append(dict(id=-1, cx=float(cx), cy=float(cy), r=float(r),
                            area=a, contour=c, bbox=(x,y,w,h)))
    targets.sort(key=lambda t: t["r"], reverse=True)
    for i, t in enumerate(targets): t["id"] = i
    return targets

def detect_tele_subpixel(tele_bgr):
    mask = _hsv_mask_red(tele_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < MIN_AREA or a > MAX_AREA: continue
        ci = _circularity(c)
        if ci < MIN_CIRCULARITY: continue
        (cx0, cy0), r0 = cv2.minEnclosingCircle(c)
        cand.append((a, ci, float(cx0), float(cy0), float(r0), c))
    cand.sort(key=lambda x: (-x[0], -x[1]))
    cand = _filter_candidates(cand)[:MAX_TARGETS_PER_IMAGE]
    if not cand: return []
    L = _red_likelihood(tele_bgr).astype(np.float64)
    targets = []
    for i, (area, ci, cx0, cy0, r0, _) in enumerate(cand):
        best = _pick_best_alpha(L, cx0, cy0, r0, area)
        if best is None: continue
        targets.append(dict(id=i, cx=best["cx"], cy=best["cy"], r=best["r"],
                            pts=best["pts"], std_cx=float(best["std"][0]),
                            std_cy=float(best["std"][1]), std_r=float(best["std"][2]),
                            alpha=best["alpha"], coverage=best["coverage"],
                            cx0=cx0, cy0=cy0, r0=r0))
    return targets


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Registration helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_gray(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 1.0)
    return cv2.createCLAHE(2.0, (8,8)).apply(g)

def _grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _compute_psr(resp, peak_xy, exclude_r=12):
    peak = float(resp[peak_xy[1], peak_xy[0]])
    mask = np.ones(resp.shape, bool)
    x, y = peak_xy
    mask[max(0,y-exclude_r):y+exclude_r+1, max(0,x-exclude_r):x+exclude_r+1] = False
    side = resp[mask]
    if side.size < 50: return float("nan")
    return (peak - float(side.mean())) / float(side.std() + 1e-12)

def _ecc_refine(tele_u8, wide_u8):
    I1 = tele_u8.astype(np.float32)/255.0
    I2 = wide_u8.astype(np.float32)/255.0
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(I1, I2, warp, ECC_MOTION, crit, None, 1)
        return float(cc), warp
    except cv2.error:
        return None, None

def _rmse_photometric(wide_u8, tele_u8):
    x = tele_u8.astype(np.float32)/255.0;  y = wide_u8.astype(np.float32)/255.0
    mx, my = x.mean(), y.mean();  vx = ((x-mx)**2).mean()+1e-12
    a = ((x-mx)*(y-my)).mean()/vx;  b = my - a*mx
    return float(np.sqrt(((y-(a*x+b))**2).mean()))

def _scale_grid(s_prior):
    smin = max(0.02, s_prior*(1-S_RANGE));  smax = min(0.95, s_prior*(1+S_RANGE))
    return np.linspace(smin, smax, N_SCALES).astype(np.float32).tolist()

def _best_scale_and_translation(wide_feat, tele_feat, tele_xy, wide_xy, s_prior):
    cx_t, cy_t = float(tele_xy[0]), float(tele_xy[1])
    cx_w, cy_w = float(wide_xy[0]), float(wide_xy[1])
    best = None
    sig_s = max(1e-9, s_prior*S_PRIOR_SIGMA_FRAC)
    for s in _scale_grid(s_prior):
        th = int(round(tele_feat.shape[0]*s));  tw = int(round(tele_feat.shape[1]*s))
        if th < 80 or tw < 80: continue
        tele_s = cv2.resize(tele_feat, (tw,th), interpolation=cv2.INTER_AREA)
        x0 = int(round(cx_w-s*cx_t-SEARCH_PAD));  y0 = int(round(cy_w-s*cy_t-SEARCH_PAD))
        x1 = x0+tw+2*SEARCH_PAD;  y1 = y0+th+2*SEARCH_PAD
        if x0<0 or y0<0 or x1>wide_feat.shape[1] or y1>wide_feat.shape[0]: continue
        resp = cv2.matchTemplate(wide_feat[y0:y1,x0:x1], tele_s, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(resp)
        psr = _compute_psr(resp, maxloc)
        lp  = -0.5*((s-s_prior)/sig_s)**2
        logpost = float(maxv) + W_PSR*(float(psr) if np.isfinite(psr) else 0.0) + W_PRIOR*lp
        cand = dict(s=float(s), score=float(maxv),
                    psr=float(psr) if np.isfinite(psr) else float("nan"),
                    logpost=logpost,
                    x=x0+int(maxloc[0]), y=y0+int(maxloc[1]), tw=tw, th=th)
        if best is None or cand["logpost"] > best["logpost"]: best = cand
    return best

def _compose_H(best, warp_roi):
    s = float(best["s"]);  x = float(best["x"]);  y = float(best["y"])
    W3 = np.vstack([warp_roi.astype(np.float32), [0,0,1]]).astype(np.float32)
    T3 = np.array([[1,0,x],[0,1,y],[0,0,1]], np.float32)
    S3 = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
    A3 = T3 @ W3 @ S3
    return A3.astype(np.float64)

def register(wide_bgr, tele_bgr, wide_targets, tele_targets):
    wide_gray = _to_gray(wide_bgr);  tele_gray = _to_gray(tele_bgr)
    wide_feat = _grad_mag(wide_gray);  tele_feat = _grad_mag(tele_gray)
    primary   = min(tele_targets,
                    key=lambda t: (t["cx"]-float(K_TELE[0,2]))**2 +
                                  (t["cy"]-float(K_TELE[1,2]))**2)
    tele_xy = (primary["cx"], primary["cy"])
    s_prior = S_PRIOR_DEFAULT;  best_all = None
    for w in wide_targets[:6]:
        t0 = time.time()
        best = _best_scale_and_translation(wide_feat, tele_feat,
                                           tele_xy, (w["cx"],w["cy"]), s_prior)
        if best is None: continue
        tele_s = cv2.resize(tele_gray, (best["tw"],best["th"]), cv2.INTER_AREA)
        patch  = wide_gray[best["y"]:best["y"]+best["th"], best["x"]:best["x"]+best["tw"]]
        if patch.shape != tele_s.shape: continue
        ecc_cc, warp_roi = _ecc_refine(tele_s, patch)
        if ecc_cc is None: continue
        tele_w = cv2.warpAffine(tele_s, warp_roi, (best["tw"],best["th"]),
                                flags=cv2.INTER_LINEAR, borderMode=BORDER_MODE)
        rmse = _rmse_photometric(patch, tele_w)
        score = float(ecc_cc) + W_FINAL_MATCH*float(best["score"]) - W_FINAL_RMSE*rmse
        print(f"  wide#{w['id']}: scale={best['s']:.4f}  NCC={best['score']:.4f}"
              f"  PSR={best['psr']:.1f}  ECC={ecc_cc:.4f}  RMSE={rmse:.4f}"
              f"  final={score:.4f}  ({time.time()-t0:.1f}s)")
        rec = dict(wide_id=w["id"], best=best, ecc_cc=float(ecc_cc),
                   warp_roi=warp_roi, rmse=rmse, final_score=score)
        if best_all is None or score > best_all["final_score"]: best_all = rec
    if best_all is None: return None, None, None
    H_total = _compose_H(best_all["best"], best_all["warp_roi"])
    return H_total, best_all, tele_gray


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — 3D fusion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apply_H(H, u, v):
    p = H @ np.array([float(u), float(v), 1.0], np.float64)
    if abs(p[2]) < 1e-12: return None
    return float(p[0]/p[2]), float(p[1]/p[2])

def _unit_ray(u, v, K, D):
    pts = np.array([[[u, v]]], np.float64)
    und = cv2.undistortPoints(pts, K, D, P=None)
    x, y = float(und[0,0,0]), float(und[0,0,1])
    r = np.array([x, y, 1.0], np.float64)
    return r / (np.linalg.norm(r) + 1e-12)

def _distance_from_boundary(center_uv, boundary_pts, K, D, sphere_r):
    rc = _unit_ray(center_uv[0], center_uv[1], K, D)
    alphas = []
    for (x, y) in boundary_pts:
        rb  = _unit_ray(float(x), float(y), K, D)
        dot = float(np.clip(np.dot(rc, rb), -1, 1))
        a   = math.atan2(float(np.linalg.norm(np.cross(rc, rb))), dot)
        if a > 1e-8: alphas.append(a)
    if len(alphas) < 20: return None, None
    alpha = float(np.median(alphas))
    return float(sphere_r / (math.sin(alpha) + 1e-12)), alpha

def _project(K, P):
    X, Y, Z = float(P[0]), float(P[1]), float(P[2])
    if abs(Z) < 1e-12: return float("nan"), float("nan")
    return float(K[0,0]*X/Z + K[0,2]), float(K[1,1]*Y/Z + K[1,2])

def _yaw_pitch(u_w, v_w, K_w):
    x = (float(u_w)-float(K_w[0,2]))/float(K_w[0,0])
    y = (float(v_w)-float(K_w[1,2]))/float(K_w[1,1])
    return math.atan2(x, 1.0), math.atan2(-y, math.sqrt(1+x*x))

def _R_from_yaw_pitch(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float64)
    Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], np.float64)
    return Ry @ Rx

def fuse_3d(tele_targets, wide_targets, H_total):
    D_w = D_WIDE if USE_DISTORTION else np.zeros_like(D_WIDE)
    D_t = D_TELE if USE_DISTORTION else np.zeros_like(D_TELE)
    sphere_r = SPHERE_DIAMETER_M * 0.5
    # dual-branch rotation matrix
    axis = _apply_H(H_total, float(K_TELE[0,2]), float(K_TELE[1,2]))
    R_t2w = _R_from_yaw_pitch(*_yaw_pitch(axis[0], axis[1], K_WIDE)) if axis else None
    results = []
    for t in tele_targets:
        cx_t, cy_t = float(t["cx"]), float(t["cy"])
        pts = t.get("pts", [])
        if len(pts) < 20:
            # fall back to circle-sampled boundary
            r = float(t.get("r", 0))
            pts = [(cx_t + r*math.cos(2*math.pi*i/72),
                    cy_t + r*math.sin(2*math.pi*i/72)) for i in range(72)]
        Dm, alpha = _distance_from_boundary((cx_t, cy_t), pts, K_TELE, D_t, sphere_r)
        if Dm is None: continue
        uv = _apply_H(H_total, cx_t, cy_t)
        if uv is None: continue
        r_w  = _unit_ray(uv[0], uv[1], K_WIDE, D_w)
        P_w  = Dm * r_w
        u_proj, v_proj = _project(K_WIDE, P_w)
        dPw  = float("nan")
        if R_t2w is not None:
            r_t  = _unit_ray(cx_t, cy_t, K_TELE, D_t)
            P_wR = R_t2w @ (Dm * r_t)
            dPw  = float(np.linalg.norm(P_w - P_wR))
        print(f"  Target {t['id']}: D={Dm:.3f}m  alpha={math.degrees(alpha):.4f}°"
              f"  P=({P_w[0]:.3f},{P_w[1]:.3f},{P_w[2]:.3f})m"
              f"  dual_err={dPw:.4f}m")
        results.append(dict(tid=t["id"], Dm=Dm, alpha=alpha,
                            P_w=P_w, u_proj=u_proj, v_proj=v_proj, dPw=dPw))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

WHITE=(255,255,255); BLACK=(0,0,0); GREEN=(50,220,50)
YELLOW=(0,230,230); CYAN=(255,220,0); ORANGE=(0,130,255)

def _lbl(img, text, xy, scale=1.0, color=WHITE, th=2):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img,text,(x+1,y+1),cv2.FONT_HERSHEY_SIMPLEX,scale,BLACK,th+2,cv2.LINE_AA)
    cv2.putText(img,text,(x,  y  ),cv2.FONT_HERSHEY_SIMPLEX,scale,color,th,  cv2.LINE_AA)

def _shrink(img, maxd=1800):
    h,w = img.shape[:2];  sc = min(1.0, maxd/max(h,w))
    return cv2.resize(img,(int(w*sc),int(h*sc)),cv2.INTER_AREA) if sc<1 else img.copy()

def _save(img, name):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, _shrink(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure generators
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1(wide_bgr, wide_targets):
    print("[Fig 1] Wide detection …")
    vis = wide_bgr.copy()
    for t in wide_targets:
        cx,cy,r = int(t["cx"]),int(t["cy"]),int(t["r"])
        cv2.circle(vis,(cx,cy),r+4,GREEN,3,cv2.LINE_AA)
        cv2.circle(vis,(cx,cy),4,YELLOW,-1,cv2.LINE_AA)
        _lbl(vis,f"#{t['id']}",(cx-r,cy-r-18),0.9,YELLOW)
    _lbl(vis,f"Wide camera — {len(wide_targets)} markers detected",(30,65),1.8,WHITE,3)
    _save(vis,"01_wide_detection.jpg")


def make_fig2(tele_bgr, tele_targets):
    print("[Fig 2] Tele subpixel fit …")
    vis = tele_bgr.copy()
    for t in tele_targets:
        cx,cy,r = t["cx"],t["cy"],t["r"]
        for j,(px,py) in enumerate(t.get("pts",[])):
            if j%3==0: cv2.circle(vis,(int(round(px)),int(round(py))),2,(0,0,255),-1,cv2.LINE_AA)
        cv2.circle(vis,(int(t["cx0"]),int(t["cy0"])),int(t["r0"]),(200,200,0),2,cv2.LINE_AA)
        cv2.circle(vis,(int(round(cx)),int(round(cy))),int(round(r)),GREEN,3,cv2.LINE_AA)
        cv2.drawMarker(vis,(int(round(cx)),int(round(cy))),CYAN,cv2.MARKER_CROSS,24,2,cv2.LINE_AA)
        _lbl(vis,f"#{t['id']}  r={r:.1f}px  std_r={t['std_r']:.3f}px  cov={t['coverage']:.0%}",
             (int(cx-r),int(cy-r-26)),1.1,GREEN)
    for i,(col,txt) in enumerate([((200,200,0),"Coarse HSV"),(GREEN,"IRLS fit"),((0,0,255),"Subpixel boundary")]):
        cv2.circle(vis,(50,40+i*44),8,col,-1)
        _lbl(vis,txt,(70,48+i*44),0.9,col)
    _lbl(vis,"Tele — subpixel circle fit (IRLS + Huber)",(30,vis.shape[0]-30),1.8,WHITE,3)
    _save(vis,"02_tele_subpixel.jpg")


def make_fig3(tele_bgr, tele_targets):
    print("[Fig 3] Likelihood heatmap …")
    L  = _red_likelihood(tele_bgr)
    L8 = np.clip(L/(L.max()+1e-12)*255,0,255).astype(np.uint8)
    heat = cv2.applyColorMap(L8, cv2.COLORMAP_INFERNO)
    for t in tele_targets:
        cv2.circle(heat,(int(t["cx"]),int(t["cy"])),int(t["r"]),(0,255,200),3,cv2.LINE_AA)
    _lbl(heat,"Red-likelihood map  L = Hue × Sat × Val^γ",(20,58),1.6,WHITE,2)
    _save(heat,"03_likelihood_map.jpg")


def make_fig4(wide_bgr, tele_bgr, wide_targets, tele_targets, H_total, best_all, tele_gray):
    print("[Fig 4] Registration blend …")
    if H_total is None: print("  [SKIP]"); return
    b = best_all["best"];  tw,th_px,bx,by = b["tw"],b["th"],b["x"],b["y"]
    tele_col = cv2.resize(tele_bgr,(tw,th_px),cv2.INTER_AREA)
    tele_warp = cv2.warpAffine(tele_col, best_all["warp_roi"],(tw,th_px),
                               flags=cv2.INTER_LINEAR, borderMode=BORDER_MODE)
    vis = wide_bgr.copy()
    y0,y1 = by, min(by+th_px, vis.shape[0])
    x0,x1 = bx, min(bx+tw,    vis.shape[1])
    roi_w = vis[y0:y1,x0:x1].astype(np.float32)
    roi_t = tele_warp[:y1-y0,:x1-x0].astype(np.float32)
    vis[y0:y1,x0:x1] = np.clip(roi_w*(1-BLEND_ALPHA)+roi_t*BLEND_ALPHA,0,255).astype(np.uint8)
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,220,255),4,cv2.LINE_AA)
    _lbl(vis,f"Tele FOV  scale={b['s']:.3f}  ECC ρ={best_all['ecc_cc']:.4f}",
         (x0+8,y0+48),1.1,(0,220,255),2)
    primary = min(tele_targets, key=lambda t:(t["cx"]-float(K_TELE[0,2]))**2+(t["cy"]-float(K_TELE[1,2]))**2)
    uv = _apply_H(H_total, primary["cx"], primary["cy"])
    if uv:
        cm,vm = int(round(uv[0])),int(round(uv[1]))
        cv2.circle(vis,(cm,vm),20,GREEN,4,cv2.LINE_AA)
        cv2.drawMarker(vis,(cm,vm),GREEN,cv2.MARKER_CROSS,60,4,cv2.LINE_AA)
        _lbl(vis,"H_total(tele centre)",(cm+25,vm-12),1.0,GREEN)
    _lbl(vis,"Fig 4  Wide-Tele Registration — scale TM + ECC + H_total",(30,68),1.7,WHITE,3)
    _save(vis,"04_registration_blend.jpg")


def make_fig5(wide_bgr, wide_targets, fusion_results, H_total):
    print("[Fig 5] 3D result …")
    if not fusion_results: print("  [SKIP]"); return
    vis = wide_bgr.copy()
    for wt in wide_targets:
        cv2.circle(vis,(int(wt["cx"]),int(wt["cy"])),int(wt["r"]),(100,100,100),1,cv2.LINE_AA)
    for r in fusion_results:
        pu,pv = int(round(r["u_proj"])),int(round(r["v_proj"]))
        cv2.circle(vis,(pu,pv),22,ORANGE,4,cv2.LINE_AA)
        cv2.drawMarker(vis,(pu,pv),YELLOW,cv2.MARKER_CROSS,60,4,cv2.LINE_AA)
        P = r["P_w"]
        lines = [f"D  = {r['Dm']:.3f} m",
                 f"X  = {P[0]:.3f} m", f"Y  = {P[1]:.3f} m", f"Z  = {P[2]:.3f} m",
                 f"α  = {math.degrees(r['alpha']):.4f}°"]
        if not math.isnan(r["dPw"]): lines.append(f"dual err = {r['dPw']*1000:.1f} mm")
        bx_l,by_l = pu+30, pv-90
        ov = vis.copy()
        cv2.rectangle(ov,(bx_l-6,by_l-6),(bx_l+420,by_l+len(lines)*38+12),BLACK,-1)
        cv2.addWeighted(ov,0.55,vis,0.45,0,vis)
        for li,line in enumerate(lines):
            col = YELLOW if li==0 else (GREEN if 1<=li<=3 else (180,180,180))
            _lbl(vis,line,(bx_l,by_l+li*38),1.0,col,2)
    _lbl(vis,"Fig 5  3D Fusion:  D = R/sin(α)   P = D × undistort(H_total · p_tele)",
         (30,68),1.5,WHITE,3)
    _save(vis,"05_3d_result.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*62)
    print("  Wide-Tele 3D Localisation — Demo Visuals")
    print("="*62)

    print("\nLoading images …")
    wide_bgr = cv2.imread(WIDE_PATH)
    tele_bgr = cv2.imread(TELE_PATH)
    assert wide_bgr is not None, f"Cannot read: {WIDE_PATH}"
    assert tele_bgr is not None, f"Cannot read: {TELE_PATH}"
    print(f"  Wide: {wide_bgr.shape[1]}×{wide_bgr.shape[0]}")
    print(f"  Tele: {tele_bgr.shape[1]}×{tele_bgr.shape[0]}")

    print("\n--- Stage 1: Wide detection ---")
    wide_targets = detect_wide_targets(wide_bgr)
    print(f"  {len(wide_targets)} markers detected")
    make_fig1(wide_bgr, wide_targets)

    print("\n--- Stage 2: Tele subpixel fit ---")
    tele_targets = detect_tele_subpixel(tele_bgr)
    print(f"  {len(tele_targets)} target(s) fitted")
    if not tele_targets:
        print("[ABORT] No tele targets found."); return
    make_fig2(tele_bgr, tele_targets)
    make_fig3(tele_bgr, tele_targets)

    print("\n--- Stage 3: Registration ---")
    H_total, best_all, tele_gray = register(wide_bgr, tele_bgr, wide_targets, tele_targets)
    if H_total is None:
        print("[ABORT] Registration failed."); return
    print(f"\n  Best: wide#{best_all['wide_id']}"
          f"  scale={best_all['best']['s']:.4f}"
          f"  ECC={best_all['ecc_cc']:.4f}"
          f"  RMSE={best_all['rmse']:.4f}")
    make_fig4(wide_bgr, tele_bgr, wide_targets, tele_targets, H_total, best_all, tele_gray)

    print("\n--- Stage 4: 3D fusion ---")
    fusion_results = fuse_3d(tele_targets, wide_targets, H_total)
    make_fig5(wide_bgr, wide_targets, fusion_results, H_total)

    print(f"\n✓  All figures saved to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
