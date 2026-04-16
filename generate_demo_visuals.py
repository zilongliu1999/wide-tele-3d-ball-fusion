# -*- coding: utf-8 -*-
"""
generate_demo_visuals.py
========================
Runs the three core modules on the REAL wide + tele images and saves
annotated output images suitable for a GitHub README or portfolio.

Usage:
    python generate_demo_visuals.py

Output (written to ./demo_output/):
    01_wide_detection.jpg          – wide image with all detected balls annotated
    02_tele_subpixel.jpg           – tele image with subpixel boundary points + fitted circle
    03_likelihood_map.jpg          – red-likelihood heatmap (tele)
    04_registration_blend.jpg      – alpha-blend of tele warped onto wide (registration check)
    05_3d_result.jpg               – wide image with 3D position annotation per ball
"""

import os, math
import cv2
import numpy as np

# ── output directory ───────────────────────────────────────────────────────────
OUT_DIR = "./demo_output"
os.makedirs(OUT_DIR, exist_ok=True)

WIDE_PATH = "./Img238.jpg"
TELE_PATH = "./Img333.jpg"

# ── camera parameters (from your calibration) ─────────────────────────────────
K_WIDE = np.array([[1.2207e4, 0.0, 2.7651e3],
                   [0.0, 1.2208e4, 1.8085e3],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
K_TELE = np.array([[6.0411e4, 0.0, 1.9527e3],
                   [0.0, 6.0421e4, 1.3781e3],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
D_ZERO = np.zeros((1, 5), dtype=np.float64)   # distortion ignored for demo
SPHERE_RADIUS_M = 0.10                          # 20 cm diameter ball


# ══════════════════════════════════════════════════════════════════════════════
#  Core algorithm functions (self-contained, no external deps beyond cv2+numpy)
# ══════════════════════════════════════════════════════════════════════════════

def compute_red_likelihood(img_bgr, hue_width=22.0, v_gamma=0.5):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32)
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0
    d = np.minimum(np.abs(h), np.abs(h - 180.0))
    closeness = np.clip(1.0 - d / hue_width, 0.0, 1.0)
    return closeness * s * np.power(v, v_gamma)


def detect_balls_coarse(img_bgr, min_area=300, min_r=5):
    """HSV segmentation → list of coarse (cx, cy, r, contour)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,  60, 60), (15, 255, 255))
    m2 = cv2.inRange(hsv, (155, 60, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min_r:
            continue
        # circularity filter
        perim = cv2.arcLength(c, True) + 1e-9
        circ = 4 * math.pi * a / (perim ** 2)
        if circ < 0.45:
            continue
        results.append((float(cx), float(cy), float(r), c))
    results.sort(key=lambda x: -x[2])
    return results, mask


def _bilinear(img_f, xs, ys):
    H, W = img_f.shape
    xs = np.clip(xs, 0, W - 2 - 1e-6);  ys = np.clip(ys, 0, H - 2 - 1e-6)
    x0 = np.floor(xs).astype(int);       y0 = np.floor(ys).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wa=(x1-xs)*(y1-ys); wb=(x1-xs)*(ys-y0); wc=(xs-x0)*(y1-ys); wd=(xs-x0)*(ys-y0)
    return img_f[y0,x0]*wa + img_f[y1,x0]*wb + img_f[y0,x1]*wc + img_f[y1,x1]*wd


def robust_circle_fit(pts, cx0, cy0, r0, iters=40, hk=1.345):
    pts = np.asarray(pts, np.float64)
    c = np.array([cx0, cy0], np.float64);  r = float(r0)
    for _ in range(iters):
        dx, dy = c[0]-pts[:,0], c[1]-pts[:,1]
        d = np.hypot(dx, dy) + 1e-12;  res = d - r
        sig = 1.4826*np.median(np.abs(res-np.median(res))) + 1e-12
        t = np.abs(res)/sig;  w = np.where(t>hk, hk/t, np.ones_like(t))
        sw = np.sqrt(w)
        J = np.column_stack([dx/d, dy/d, -np.ones(len(d))])
        dlt = np.linalg.solve((J*sw[:,None]).T@(J*sw[:,None]) + 1e-6*np.eye(3),
                              (J*sw[:,None]).T@(-res*sw))
        c += dlt[:2];  r = max(float(r+dlt[2]), 1.0)
        if np.linalg.norm(dlt) < 1e-6: break
    # uncertainty
    dx, dy = c[0]-pts[:,0], c[1]-pts[:,1]
    d = np.hypot(dx,dy)+1e-12; res = d-r
    sig = 1.4826*np.median(np.abs(res-np.median(res)))+1e-12
    t = np.abs(res)/sig; w = np.where(t>hk, hk/t, np.ones_like(t))
    J = np.column_stack([dx/d, dy/d, -np.ones(len(d))])
    dof = max(len(res)-3,1)
    s2 = np.sum(w*res**2)/dof
    std = np.sqrt(np.clip(np.diag(np.linalg.pinv((J.T*w)@J)*s2), 0, np.inf))
    return float(c[0]), float(c[1]), float(r), res, std


def subpixel_boundary(img_bgr, cx0, cy0, r0,
                      theta_bins=720, band_px=None, step=0.4, alpha=0.25):
    band_px = band_px or max(15.0, 0.20*r0)
    L = compute_red_likelihood(img_bgr).astype(np.float64)
    rs = np.arange(max(1.0, r0-band_px), r0+band_px+1e-9, step)
    pts = []
    for bi in range(theta_bins):
        th = 2*math.pi*bi/theta_bins
        ct, st = math.cos(th), math.sin(th)
        xs, ys = cx0+rs*ct, cy0+rs*st
        prof = _bilinear(L, xs, ys)
        p32 = prof.astype(np.float32).reshape(-1,1)
        prof = cv2.GaussianBlur(p32,(1,9),1.5).reshape(-1).astype(np.float64)
        kf = max(4, int(0.10*len(prof)))
        Lin, Lout = float(np.median(prof[:kf])), float(np.median(prof[-kf:]))
        if Lin < 0.05 or Lout > 0.06 or (Lin-Lout) < 0.025: continue
        g = (prof[1:]-prof[:-1])/step
        idx = int(np.argmin(g))
        if -g[idx] < 0.008: continue
        delta = 0.0
        if 1 <= idx < len(g)-1:
            ym1,y0,yp1 = -g[idx-1],-g[idx],-g[idx+1]
            denom = ym1-2*y0+yp1
            if abs(denom) > 1e-12: delta = float(np.clip(0.5*(ym1-yp1)/denom,-1,1))
        r_edge = rs[idx]+(0.5+delta)*step
        tval = Lout+alpha*(Lin-Lout)
        for i in range(max(0,idx-14), min(len(prof)-2,idx+14)+1):
            if prof[i]>=tval>prof[i+1]:
                frac=(prof[i]-tval)/(prof[i]-prof[i+1]+1e-12)
                r_edge=rs[i]+frac*step; break
        pts.append((cx0+r_edge*ct, cy0+r_edge*st))
    return pts


def estimate_distance(center_uv, boundary_pts, K, sphere_radius_m):
    def ray(u, v):
        p = np.array([[[u,v]]], np.float64)
        und = cv2.undistortPoints(p, K, D_ZERO, P=None)
        x,y = float(und[0,0,0]), float(und[0,0,1])
        r = np.array([x,y,1.0]); return r/np.linalg.norm(r)
    rc = ray(*center_uv)
    alphas = []
    for (x,y) in boundary_pts:
        rb = ray(float(x), float(y))
        dot = float(np.clip(np.dot(rc,rb),-1,1))
        sinv = float(np.linalg.norm(np.cross(rc,rb)))
        a = math.atan2(sinv, dot)
        if a > 1e-8: alphas.append(a)
    if len(alphas) < 20: return None, None
    alpha = float(np.median(alphas))
    return float(sphere_radius_m/(math.sin(alpha)+1e-12)), alpha


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

ORANGE = (0, 100, 255)   # BGR
GREEN  = (50, 220, 50)
CYAN   = (255, 220, 0)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 230, 230)

def put_label(img, text, xy, scale=1.0, color=WHITE, thickness=2):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def scale_for_display(img, max_dim=1600):
    h, w = img.shape[:2]
    sc = min(1.0, max_dim / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    return img, sc


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Wide image: all detected balls annotated
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1_wide_detection(wide_bgr):
    print("[Fig 1] Wide image ball detection …")
    balls, mask = detect_balls_coarse(wide_bgr, min_area=500, min_r=8)
    vis = wide_bgr.copy()
    for i, (cx, cy, r, cnt) in enumerate(balls):
        cv2.circle(vis,(int(cx),int(cy)),int(r+4),(0,200,0),3,cv2.LINE_AA)
        cv2.circle(vis,(int(cx),int(cy)),3,YELLOW,-1,cv2.LINE_AA)
        put_label(vis, f"#{i}  r={r:.0f}px",
                  (cx-r, cy-r-18), scale=1.0, color=YELLOW, thickness=2)
    put_label(vis, f"Wide camera — {len(balls)} balls detected",
              (30, 60), scale=2.0, color=WHITE, thickness=3)
    out, _ = scale_for_display(vis, 1600)
    path = os.path.join(OUT_DIR, "01_wide_detection.jpg")
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}  ({len(balls)} balls)")
    return balls


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 & 3 — Tele image: subpixel boundary + likelihood map
# ══════════════════════════════════════════════════════════════════════════════

def make_fig2_tele_subpixel(tele_bgr):
    print("[Fig 2] Tele subpixel detection …")
    balls, _ = detect_balls_coarse(tele_bgr, min_area=2000, min_r=30)
    if not balls:
        print("  [WARN] No balls found in tele image")
        return []

    vis = tele_bgr.copy()
    results = []

    for i, (cx0, cy0, r0, cnt) in enumerate(balls[:3]):   # show up to 3
        print(f"  Ball #{i}: coarse cx={cx0:.0f} cy={cy0:.0f} r={r0:.0f}")
        pts = subpixel_boundary(tele_bgr, cx0, cy0, r0)
        if len(pts) < 60:
            print(f"    [WARN] only {len(pts)} boundary points — skipping fit")
            continue
        cx, cy, r, _, std = robust_circle_fit(pts, cx0, cy0, r0)
        print(f"    Fitted: cx={cx:.2f} cy={cy:.2f} r={r:.2f}  "
              f"std=[{std[0]:.3f},{std[1]:.3f},{std[2]:.3f}]")

        # draw boundary dots (every 3rd)
        for j, (px, py) in enumerate(pts):
            if j % 3 == 0:
                cv2.circle(vis,(int(round(px)),int(round(py))),2,(0,0,255),-1,cv2.LINE_AA)

        # coarse circle
        cv2.circle(vis,(int(cx0),int(cy0)),int(r0),(200,200,0),2,cv2.LINE_AA)
        # fitted circle
        cv2.circle(vis,(int(round(cx)),int(round(cy))),int(round(r)),GREEN,3,cv2.LINE_AA)
        # centre cross
        cv2.drawMarker(vis,(int(round(cx)),int(round(cy))),CYAN,
                       cv2.MARKER_CROSS,20,2,cv2.LINE_AA)

        put_label(vis,
                  f"#{i}  r={r:.1f}px  std_r={std[2]:.2f}px  n={len(pts)}pts",
                  (int(cx-r), int(cy-r-22)), scale=1.1, color=GREEN)
        results.append(dict(cx=cx,cy=cy,r=r,std=std,pts=pts,cx0=cx0,cy0=cy0,r0=r0))

    # legend
    cv2.circle(vis,(50,40),8,(200,200,0),-1); put_label(vis,"Coarse HSV",(70,48),0.9,color=(200,200,0))
    cv2.circle(vis,(50,80),8,GREEN,-1);       put_label(vis,"IRLS fit",(70,88),0.9,color=GREEN)
    cv2.circle(vis,(50,120),4,(0,0,255),-1);  put_label(vis,"Subpixel boundary pts",(70,128),0.9,color=(0,0,255))
    put_label(vis,"Tele camera — subpixel circle fit",(30,vis.shape[0]-30),
              scale=1.8, color=WHITE, thickness=3)

    out, _ = scale_for_display(vis, 1600)
    path = os.path.join(OUT_DIR, "02_tele_subpixel.jpg")
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}")
    return results


def make_fig3_likelihood(tele_bgr):
    print("[Fig 3] Likelihood heatmap …")
    L = compute_red_likelihood(tele_bgr)
    # Normalise to uint8 and apply colour map
    L8 = np.clip(L / (L.max()+1e-12) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(L8, cv2.COLORMAP_INFERNO)
    # overlay contour of balls for reference
    balls, _ = detect_balls_coarse(tele_bgr, min_area=2000, min_r=30)
    for (cx,cy,r,cnt) in balls[:3]:
        cv2.circle(heat,(int(cx),int(cy)),int(r),(0,255,200),3,cv2.LINE_AA)
    put_label(heat,"Red-likelihood map  L = Hue × Sat × Val^γ",
              (20, 55), scale=1.6, color=WHITE, thickness=2)
    out, _ = scale_for_display(heat, 1600)
    path = os.path.join(OUT_DIR, "03_likelihood_map.jpg")
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Registration: warp tele onto wide (alpha blend)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig4_registration(wide_bgr, tele_bgr, wide_balls, tele_results):
    print("[Fig 4] Registration blend …")
    if not wide_balls or not tele_results:
        print("  [SKIP] Need at least one ball in each image")
        return None

    Hw, Ww = wide_bgr.shape[:2]
    Ht, Wt = tele_bgr.shape[:2]

    # Use the largest tele ball and best-matching wide ball
    tr = tele_results[0]
    # Estimate scale: tele focal length / wide focal length ≈ 60411/12207 ≈ 4.95
    # so tele covers ~1/5 of the wide FOV
    s = float(K_TELE[0,0]) / float(K_WIDE[0,0]) * 0.20   # empirical display scale

    tw = max(60, int(Wt * s))
    th = max(60, int(Ht * s))
    tele_small = cv2.resize(tele_bgr, (tw, th), interpolation=cv2.INTER_AREA)

    # Place tele thumbnail centred on the corresponding wide ball
    wb = wide_balls[0]
    cx_w, cy_w = int(wb[0]), int(wb[1])
    x0 = max(0, cx_w - tw//2);  y0 = max(0, cy_w - th//2)
    x1 = min(Ww, x0+tw);        y1 = min(Hw, y0+th)
    tw2, th2 = x1-x0, y1-y0

    blend = wide_bgr.copy().astype(np.float32)
    tele_roi = tele_small[:th2, :tw2].astype(np.float32)
    blend[y0:y1, x0:x1] = blend[y0:y1, x0:x1]*0.50 + tele_roi*0.50
    blend = np.clip(blend, 0, 255).astype(np.uint8)

    # Annotate
    cv2.rectangle(blend,(x0,y0),(x1,y1),(0,220,255),3,cv2.LINE_AA)
    put_label(blend,"Tele FOV (warped onto wide)",(x0+6,y0+32),1.0,color=(0,220,255))
    cv2.circle(blend,(cx_w,cy_w),int(wb[2]+4),GREEN,3,cv2.LINE_AA)
    put_label(blend,"Wide-Tele Registration — alpha blend (α=0.5)",
              (30,60),scale=2.0,color=WHITE,thickness=3)
    # draw an arrow from tele overlay centre to wide ball
    cv2.arrowedLine(blend,(x0+tw2//2, y0+th2//2),(cx_w,cy_w),YELLOW,3,
                    cv2.LINE_AA, tipLength=0.03)

    out, _ = scale_for_display(blend, 1600)
    path = os.path.join(OUT_DIR, "04_registration_blend.jpg")
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}")
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — 3D result: distance estimate projected onto wide
# ══════════════════════════════════════════════════════════════════════════════

def make_fig5_3d_result(wide_bgr, tele_results, wide_balls):
    print("[Fig 5] 3D distance estimation …")
    vis = wide_bgr.copy()

    for i, tr in enumerate(tele_results):
        D_m, alpha = estimate_distance(
            (tr["cx"], tr["cy"]), tr["pts"], K_TELE, SPHERE_RADIUS_M)
        if D_m is None:
            print(f"  Ball #{i}: distance estimation failed")
            continue
        print(f"  Ball #{i}: D = {D_m:.3f} m  alpha = {math.degrees(alpha):.4f}°")

        # Corresponding wide ball (by index)
        if i < len(wide_balls):
            cx_w, cy_w, r_w = wide_balls[i][0], wide_balls[i][1], wide_balls[i][2]
        else:
            cx_w, cy_w, r_w = wide_balls[0][0], wide_balls[0][1], wide_balls[0][2]

        # Draw result on wide image
        cv2.circle(vis,(int(cx_w),int(cy_w)),int(r_w+6),ORANGE,3,cv2.LINE_AA)
        cv2.circle(vis,(int(cx_w),int(cy_w)),5,YELLOW,-1,cv2.LINE_AA)

        label = f"D={D_m:.2f}m   alpha={math.degrees(alpha):.3f}deg"
        put_label(vis, label,
                  (int(cx_w - r_w), int(cy_w - r_w - 26)),
                  scale=1.1, color=YELLOW, thickness=2)

        # also show uncertainty
        if tr.get("std") is not None:
            std_r = tr["std"][2]
            # propagate: dD/dr_px ≈ -D / (r_px * tan(alpha))
            r_px = tr["r"]
            dD = D_m / (r_px + 1e-9) * std_r
            put_label(vis, f"std_r={std_r:.2f}px  -> stdD~{dD:.3f}m",
                      (int(cx_w - r_w), int(cy_w - r_w - 56)),
                      scale=0.9, color=(200,200,200), thickness=2)

    put_label(vis, "3D Localisation:  D = R / sin(alpha)",
              (30, 60), scale=2.0, color=WHITE, thickness=3)
    put_label(vis, "distance estimated from tele angular radius",
              (30, 110), scale=1.4, color=(200,200,200), thickness=2)

    out, _ = scale_for_display(vis, 1600)
    path = os.path.join(OUT_DIR, "05_3d_result.jpg")
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading images …")
    wide_bgr = cv2.imread(WIDE_PATH)
    tele_bgr = cv2.imread(TELE_PATH)
    assert wide_bgr is not None, f"Cannot read {WIDE_PATH}"
    assert tele_bgr is not None, f"Cannot read {TELE_PATH}"
    print(f"  Wide: {wide_bgr.shape}   Tele: {tele_bgr.shape}")

    wide_balls   = make_fig1_wide_detection(wide_bgr)
    tele_results = make_fig2_tele_subpixel(tele_bgr)
    make_fig3_likelihood(tele_bgr)
    make_fig4_registration(wide_bgr, tele_bgr, wide_balls, tele_results)
    make_fig5_3d_result(wide_bgr, tele_results, wide_balls)

    print(f"\n✓  All figures saved to  {os.path.abspath(OUT_DIR)}/")
    print("   01_wide_detection.jpg      — wide image, all balls annotated")
    print("   02_tele_subpixel.jpg       — subpixel boundary + IRLS fit")
    print("   03_likelihood_map.jpg      — red-likelihood heatmap")
    print("   04_registration_blend.jpg  — tele warped onto wide")
    print("   05_3d_result.jpg           — 3D distance per ball")


if __name__ == "__main__":
    main()
