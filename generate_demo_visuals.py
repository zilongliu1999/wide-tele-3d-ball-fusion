# -*- coding: utf-8 -*-
"""
generate_demo_visuals.py
========================
Runs the FULL wide-tele 3D localisation pipeline on real field images and
saves annotated figures for the GitHub README.

Every figure is produced by the same algorithm described in the README —
there is no approximation or shortcut:

  Fig 1  Wide-angle detection        detect_wide_red_targets()
  Fig 2  Tele subpixel fit           process_tele_subpixel()  [IRLS + alpha-sweep]
  Fig 3  Red-likelihood heatmap      compute_red_likelihood()
  Fig 4  Wide-tele registration      best_scale_and_translation() + ecc_refine()
                                     + compose_affine()  →  real H_total
  Fig 5  3D fusion result            compute_distance_from_boundary_points()
                                     + undistort_point_to_unit_ray()
                                     + apply_H_to_point()  →  P = D × r_wide

Usage
-----
Place this file alongside the original pipeline script and both input images,
then run:

    python generate_demo_visuals.py

Requirements
------------
    pip install opencv-python numpy
    # plus the original pipeline file:
    wide_tele_ballcenter_3d_fusion_v8_dualbranch_mainDw_rw.py

Output
------
All figures are written to ./demo_output/
"""

import importlib.util, os, math, time
import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
PIPELINE_FILE = "./wide_tele_ballcenter_3d_fusion_v8_dualbranch_mainDw_rw.py"
WIDE_PATH     = "./Img238.jpg"
TELE_PATH     = "./Img333.jpg"
OUT_DIR       = "./demo_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load original pipeline as a module ────────────────────────────────────────
spec = importlib.util.spec_from_file_location("pipeline", PIPELINE_FILE)
pl   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


# ── colour palette ────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0  )
GREEN  = (50,  220, 50 )
YELLOW = (0,   230, 230)
CYAN   = (255, 220, 0  )
ORANGE = (0,   130, 255)


def label(img, text, xy, scale=1.0, color=WHITE, th=2):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, th+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, th,   cv2.LINE_AA)


def shrink(img, maxd=1800):
    h, w = img.shape[:2]
    sc = min(1.0, maxd / max(h, w))
    return cv2.resize(img, (int(w*sc), int(h*sc)), cv2.INTER_AREA) if sc < 1 else img.copy()


def save(img, name):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, shrink(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Wide-angle detection
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1(wide_bgr, wide_targets):
    """Annotate all detected markers on the wide image."""
    print("[Fig 1] Wide detection …")
    vis = wide_bgr.copy()
    for t in wide_targets:
        cx, cy, r = int(t["cx"]), int(t["cy"]), int(t["r"])
        cv2.circle(vis, (cx, cy), r + 4, GREEN, 3, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 4, YELLOW, -1, cv2.LINE_AA)
        label(vis, f"#{t['id']}", (cx - r, cy - r - 18), 0.9, YELLOW)
    label(vis, f"Wide camera — {len(wide_targets)} markers detected",
          (30, 65), 1.8, WHITE, 3)
    save(vis, "01_wide_detection.jpg")
    return wide_targets


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Tele subpixel fit  (uses pl.process_tele_subpixel)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig2(tele_bgr):
    """Run IRLS subpixel detection on tele image; annotate results."""
    print("[Fig 2] Tele subpixel detection …")
    subdir = os.path.join(OUT_DIR, "_tele_work")
    os.makedirs(subdir, exist_ok=True)
    targets = pl.process_tele_subpixel(tele_bgr, out_dir=subdir)
    print(f"  {len(targets)} target(s) found")

    vis = tele_bgr.copy()
    for t in targets:
        cx, cy, r = t["cx"], t["cy"], t["r"]
        pts = t.get("pts", [])
        # boundary dots every 3rd point
        for j, (px, py) in enumerate(pts):
            if j % 3 == 0:
                cv2.circle(vis, (int(round(px)), int(round(py))),
                           2, (0, 0, 255), -1, cv2.LINE_AA)
        # coarse HSV circle
        cv2.circle(vis, (int(t["cx0"]), int(t["cy0"])), int(t["r0"]),
                   (200, 200, 0), 2, cv2.LINE_AA)
        # IRLS fit
        cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)),
                   GREEN, 3, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(round(cx)), int(round(cy))),
                       CYAN, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        std_r = t.get("std_r", 0.0)
        label(vis,
              f"#{t['id']}  r={r:.1f}px  std_r={std_r:.3f}px"
              f"  cov={t.get('coverage',0):.0%}",
              (int(cx - r), int(cy - r - 26)), 1.1, GREEN)

    # legend
    for i, (col, txt) in enumerate([
            ((200, 200, 0), "Coarse HSV"),
            (GREEN,         "IRLS fit"),
            ((0, 0, 255),   "Subpixel boundary")]):
        yy = 40 + i * 44
        cv2.circle(vis, (50, yy), 8, col, -1)
        label(vis, txt, (70, yy + 8), 0.9, col)
    label(vis, "Tele — subpixel circle fit (IRLS + Huber)",
          (30, vis.shape[0] - 30), 1.8, WHITE, 3)
    save(vis, "02_tele_subpixel.jpg")
    return targets


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Red-likelihood heatmap
# ══════════════════════════════════════════════════════════════════════════════

def make_fig3(tele_bgr, tele_targets):
    print("[Fig 3] Likelihood heatmap …")
    L  = pl.compute_red_likelihood(tele_bgr,
                                   hue_width=pl.HUE_WIDTH,
                                   v_gamma=pl.V_GAMMA)
    L8 = np.clip(L / (L.max() + 1e-12) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(L8, cv2.COLORMAP_INFERNO)
    for t in tele_targets:
        cv2.circle(heat, (int(t["cx"]), int(t["cy"])), int(t["r"]),
                   (0, 255, 200), 3, cv2.LINE_AA)
    label(heat, "Red-likelihood map  L = Hue × Sat × Val^γ",
          (20, 58), 1.6, WHITE, 2)
    save(heat, "03_likelihood_map.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Wide-tele registration (real multi-scale TM + ECC + H_total)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig4(wide_bgr, tele_bgr, wide_targets, tele_targets):
    """
    Runs best_scale_and_translation → ecc_refine → compose_affine
    exactly as the production pipeline does. Returns H_total.
    """
    print("[Fig 4] Registration (multi-scale TM + ECC) …")

    wide_gray = pl.to_gray_u8(wide_bgr)
    tele_gray = pl.to_gray_u8(tele_bgr)
    wide_feat = pl.grad_mag_u8(wide_gray)
    tele_feat = pl.grad_mag_u8(tele_gray)

    primary = pl.pick_primary_target(tele_targets,
                                     float(pl.K_TELE[0, 2]),
                                     float(pl.K_TELE[1, 2]))
    tele_xy = (primary["cx"], primary["cy"])

    s_prior  = float(pl.S_PRIOR_DEFAULT)
    best_all = None

    for w in wide_targets[:6]:           # top-6 by size, same as production
        wid  = int(w["id"])
        cxw, cyw = float(w["cx"]), float(w["cy"])

        t0   = time.time()
        best = pl.best_scale_and_translation(
            wide_feat, tele_feat,
            tele_primary_xy=tele_xy,
            wide_cand_xy=(cxw, cyw),
            s_prior=s_prior,
            scan_csv_path=None,
            best_heatmap_path=None)
        if best is None:
            print(f"  wide#{wid}: TM failed ({time.time()-t0:.1f}s)")
            continue

        tele_s = cv2.resize(tele_gray, (best["tw"], best["th"]),
                            interpolation=cv2.INTER_AREA)
        patch  = wide_gray[best["y"]:best["y"]+best["th"],
                           best["x"]:best["x"]+best["tw"]]
        if patch.shape != tele_s.shape:
            continue

        ecc_cc, warp_roi = pl.ecc_refine(tele_s, patch)
        if ecc_cc is None:
            print(f"  wide#{wid}: ECC failed")
            continue

        tele_warp = cv2.warpAffine(tele_s, warp_roi,
                                   (best["tw"], best["th"]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=pl.BORDER_MODE)
        rmse, _, _ = pl.rmse_photometric(patch, tele_warp)
        score = (float(ecc_cc)
                 + pl.W_FINAL_MATCH * float(best["score"])
                 - pl.W_FINAL_RMSE  * float(rmse))

        print(f"  wide#{wid}: scale={best['s']:.4f}  "
              f"NCC={best['score']:.4f}  PSR={best['psr']:.1f}  "
              f"ECC={ecc_cc:.4f}  RMSE={rmse:.4f}  "
              f"final={score:.4f}  ({time.time()-t0:.1f}s)")

        rec = dict(wide_id=wid, best=best, ecc_cc=float(ecc_cc),
                   warp_roi=warp_roi, rmse=float(rmse), final_score=float(score))
        if best_all is None or score > best_all["final_score"]:
            best_all = rec

    if best_all is None:
        print("  [WARN] Registration failed — skipping Fig 4 / Fig 5")
        return None

    _, H_total = pl.compose_affine(best_all["best"], best_all["warp_roi"])
    best_s     = best_all["best"]["s"]
    print(f"\n  Best: wide#{best_all['wide_id']}  "
          f"scale={best_s:.4f}  ECC={best_all['ecc_cc']:.4f}  "
          f"RMSE={best_all['rmse']:.4f}")

    # ── visualise: warp full-colour tele patch onto wide ──────────────────────
    b     = best_all["best"]
    tw, th_px = b["tw"], b["th"]
    bx, by    = b["x"],  b["y"]

    tele_col_s = cv2.resize(tele_bgr, (tw, th_px), interpolation=cv2.INTER_AREA)
    tele_warp_col = cv2.warpAffine(tele_col_s, best_all["warp_roi"],
                                   (tw, th_px),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=pl.BORDER_MODE)

    vis = wide_bgr.copy()
    y0, y1 = by, min(by + th_px, vis.shape[0])
    x0, x1 = bx, min(bx + tw,    vis.shape[1])
    th2, tw2 = y1 - y0, x1 - x0

    roi_w = vis[y0:y1, x0:x1].astype(np.float32)
    roi_t = tele_warp_col[:th2, :tw2].astype(np.float32)
    vis[y0:y1, x0:x1] = np.clip(
        roi_w * (1 - pl.BLEND_ALPHA) + roi_t * pl.BLEND_ALPHA,
        0, 255).astype(np.uint8)

    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 4, cv2.LINE_AA)
    label(vis, f"Tele FOV  scale={best_s:.3f}  ECC ρ={best_all['ecc_cc']:.4f}",
          (x0 + 8, y0 + 48), 1.1, (0, 220, 255), 2)

    # project primary tele target onto wide via H_total
    uv = pl.apply_H_to_point(H_total, primary["cx"], primary["cy"])
    if uv:
        cx_m, cy_m = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(vis,    (cx_m, cy_m), 20, GREEN, 4, cv2.LINE_AA)
        cv2.drawMarker(vis, (cx_m, cy_m), GREEN,
                       cv2.MARKER_CROSS, 60, 4, cv2.LINE_AA)
        label(vis, "H_total(tele centre)", (cx_m + 25, cy_m - 12), 1.0, GREEN)

    label(vis,
          "Fig 4  Wide-Tele Registration — scale TM + ECC + H_total",
          (30, 68), 1.7, WHITE, 3)
    save(vis, "04_registration_blend.jpg")
    return H_total


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — 3D fusion  (real H_total + undistort + D × r_wide)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig5(wide_bgr, tele_targets, wide_targets, H_total):
    """
    For each tele target:
      1. compute_distance_from_boundary_points  →  D, alpha
      2. apply_H_to_point(H_total, cx_t, cy_t)  →  wide pixel
      3. undistort_point_to_unit_ray             →  r_wide
      4. P = D * r_wide
    Same code path as the production pipeline's main().
    """
    print("[Fig 5] 3D fusion …")
    if H_total is None:
        print("  [SKIP] No H_total available")
        return

    D_w = pl.D_WIDE if pl.USE_DISTORTION else np.zeros_like(pl.D_WIDE)
    D_t = pl.D_TELE if pl.USE_DISTORTION else np.zeros_like(pl.D_TELE)
    sphere_r = float(pl.SPHERE_DIAMETER_M) * 0.5

    vis = wide_bgr.copy()
    # draw all wide targets lightly for context
    for wt in wide_targets:
        cv2.circle(vis, (int(wt["cx"]), int(wt["cy"])), int(wt["r"]),
                   (100, 100, 100), 1, cv2.LINE_AA)

    # yaw/pitch of tele axis in wide frame (for dual-branch check)
    u_ax, v_ax = float(pl.K_TELE[0, 2]), float(pl.K_TELE[1, 2])
    axis = pl.apply_H_to_point(H_total, u_ax, v_ax)
    if axis:
        yaw, pitch = pl.yaw_pitch_from_wide_pixel(axis[0], axis[1], pl.K_WIDE)
        R_t2w      = pl.R_from_yaw_pitch(yaw, pitch)
    else:
        R_t2w = None

    for t in tele_targets:
        cx_t, cy_t = float(t["cx"]), float(t["cy"])
        r_fit      = float(t.get("r", 0.0))

        # boundary points: prefer real subpixel pts, fall back to circle sample
        pts = t.get("pts", [])
        if len(pts) < 20:
            pts = pl.sample_circle_boundary(cx_t, cy_t, r_fit, n=72)

        # 1 — depth from tele angular radius
        Dm, alpha = pl.compute_distance_from_boundary_points(
            tele_center_uv=(cx_t, cy_t),
            boundary_pts_uv=pts,
            K=pl.K_TELE, D=D_t,
            sphere_radius_m=sphere_r)
        if Dm is None:
            Dm, alpha = pl.compute_distance_from_radius(r_fit, pl.K_TELE, sphere_r)
        if Dm is None:
            print(f"  Target {t['id']}: depth estimation failed")
            continue

        # 2 — map tele centre to wide pixel via H_total
        uv = pl.apply_H_to_point(H_total, cx_t, cy_t)
        if uv is None:
            continue
        u_w, v_w = uv

        # 3 — wide unit ray
        r_w = pl.undistort_point_to_unit_ray(u_w, v_w, pl.K_WIDE, D_w)

        # 4 — 3D position in wide frame
        P_w    = Dm * r_w
        u_proj, v_proj = pl.project_point(pl.K_WIDE, P_w)

        # dual-branch check
        dPw = float("nan")
        if R_t2w is not None:
            r_t  = pl.undistort_point_to_unit_ray(cx_t, cy_t, pl.K_TELE, D_t)
            P_wR = R_t2w @ (Dm * r_t)
            dPw  = float(np.linalg.norm(P_w - P_wR))

        print(f"  Target {t['id']}: D={Dm:.3f}m  "
              f"alpha={math.degrees(alpha):.4f}°  "
              f"P=({P_w[0]:.3f},{P_w[1]:.3f},{P_w[2]:.3f})m  "
              f"dual_err={dPw:.4f}m")

        # annotate on wide image at projected pixel
        pu, pv = int(round(u_proj)), int(round(v_proj))
        cv2.circle(vis,    (pu, pv), 22, ORANGE, 4, cv2.LINE_AA)
        cv2.drawMarker(vis, (pu, pv), YELLOW,
                       cv2.MARKER_CROSS, 60, 4, cv2.LINE_AA)

        lines = [
            f"D  = {Dm:.3f} m",
            f"X  = {P_w[0]:.3f} m",
            f"Y  = {P_w[1]:.3f} m",
            f"Z  = {P_w[2]:.3f} m",
            f"α  = {math.degrees(alpha):.4f}°",
        ]
        if not math.isnan(dPw):
            lines.append(f"dual err = {dPw*1000:.1f} mm")

        bx_l, by_l = pu + 30, pv - 90
        ov = vis.copy()
        cv2.rectangle(ov, (bx_l - 6, by_l - 6),
                      (bx_l + 420, by_l + len(lines)*38 + 12),
                      BLACK, -1)
        cv2.addWeighted(ov, 0.55, vis, 0.45, 0, vis)
        for li, line in enumerate(lines):
            col = YELLOW if li == 0 else (GREEN if 1 <= li <= 3 else (180, 180, 180))
            label(vis, line, (bx_l, by_l + li*38), 1.0, col, 2)

    label(vis,
          "Fig 5  3D Fusion:  D = R/sin(α)   P = D × undistort(H_total · p_tele)",
          (30, 68), 1.5, WHITE, 3)
    save(vis, "05_3d_result.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Wide-Tele 3D Localisation — Demo Visuals")
    print("=" * 62)

    print("\nLoading images …")
    wide_bgr = cv2.imread(WIDE_PATH)
    tele_bgr = cv2.imread(TELE_PATH)
    assert wide_bgr is not None, f"Cannot read: {WIDE_PATH}"
    assert tele_bgr is not None, f"Cannot read: {TELE_PATH}"
    print(f"  Wide : {wide_bgr.shape[1]}×{wide_bgr.shape[0]}")
    print(f"  Tele : {tele_bgr.shape[1]}×{tele_bgr.shape[0]}")

    print("\n--- Stage 1: Wide detection ---")
    wide_targets = pl.detect_wide_red_targets(wide_bgr)
    make_fig1(wide_bgr, wide_targets)

    print("\n--- Stage 2: Tele subpixel fit ---")
    tele_targets = make_fig2(tele_bgr)
    make_fig3(tele_bgr, tele_targets)

    if not tele_targets:
        print("[ABORT] No tele targets found — cannot proceed to registration.")
        return

    print("\n--- Stage 3: Registration ---")
    H_total = make_fig4(wide_bgr, tele_bgr, wide_targets, tele_targets)

    print("\n--- Stage 4: 3D fusion ---")
    make_fig5(wide_bgr, tele_targets, wide_targets, H_total)

    print(f"\n✓  All figures saved to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
