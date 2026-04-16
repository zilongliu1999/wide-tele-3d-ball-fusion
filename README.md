# Wide-Tele 3D Ball Localisation

A minimal, self-contained demo of the core algorithms behind a dual-camera
3D ball centre localisation system — designed as a portfolio piece for
CV / algorithm engineering roles.

---

## What this demonstrates

| Module | Technique | Why it's interesting |
|---|---|---|
| **Subpixel Detection** | Radial profile sampling + IRLS circle fit | Sub-pixel accuracy via Huber-weighted least squares |
| **Image Registration** | Multi-scale template matching + ECC refinement | PSR scoring + log-Gaussian scale prior for robust matching |
| **3D Fusion** | Angular-radius distance (tele) × wide camera ray | Avoids explicit baseline estimation; robust to zoom/rotation |

---

## Background

Standard stereo triangulation requires a precisely calibrated baseline.
This pipeline drops that requirement by exploiting the **known physical size**
of the ball:

```
Depth D = R_sphere / sin(α)
```

where `α` is the median angular radius measured from **subpixel boundary
points** on the tele image. The direction vector is then taken from the
wide camera (stable, wide field of view), giving a 3D position in the
wide camera frame.

```
P_wide = D × r_wide
```

---

## Run the demo

```bash
pip install opencv-python numpy
python ball_3d_localization_demo.py
```

All three modules run on **synthetic images** generated at runtime — no
dataset required. Expected output:

```
[Module 1] Subpixel ball detection on synthetic image …
  Ground truth : cx=318.7  cy=241.3  r=58.5
  Detected     : cx=318.72 ± 0.08  cy=241.31 ± 0.07  r=58.48 ± 0.05  (n_pts=287)
  Centre error : 0.021 px

[Module 2] Wide-tele registration on synthetic pair …
  Best scale   : 0.250  (target ≈ 0.250)
  NCC          : 0.8941
  ECC ρ        : 0.9712

[Module 3] 3D fusion with known camera parameters …
  Ground truth : Z = 3.000 m
  Estimated    : X=0.0000  Y=0.0000  Z=3.0002 m
  Depth error  : 0.02 cm
```

---

## Code structure

```
ball_3d_localization_demo.py
├── Module 1 — Subpixel Ball Detection
│   ├── compute_red_likelihood()       HSV likelihood map
│   ├── detect_ball_subpixel()         Full pipeline: segment → sample → fit
│   └── robust_circle_fit()            IRLS circle fit with Huber weights
│
├── Module 2 — Image Registration
│   ├── register_tele_to_wide()        Multi-scale TM + PSR + ECC
│   └── _compute_psr()                 Peak-to-Sidelobe Ratio
│
├── Module 3 — 3D Fusion
│   ├── estimate_distance_from_angular_radius()   Depth from known sphere size
│   ├── undistort_to_unit_ray()                   Pixel → unit ray (with distortion)
│   └── fuse_3d_position()                        Full dual-branch fusion
│
└── demo()                             Smoke test on synthetic data
```

---

## Dependencies

- Python ≥ 3.10
- `opencv-python` ≥ 4.5
- `numpy` ≥ 1.22

---

## Algorithm notes

### Subpixel detection
Boundary points are extracted at `N` angles by scanning a radial profile
through a **red-likelihood map** (hue × saturation × value^γ).
Each edge position is refined to sub-pixel resolution via:
1. Parabola fit on the gradient peak
2. Cross-threshold interpolation in a local window

The circle is then fit with **IRLS** (Huber M-estimator), which suppresses
outlier boundary points caused by specular highlights or partial occlusion.

### Multi-scale registration
The tele-to-wide scale ratio is unknown at runtime. A PSR-scored template
search over a scale grid finds the best scale, then **ECC** (Enhanced
Correlation Coefficient) provides sub-pixel Euclidean alignment at that scale.
A log-Gaussian prior on scale prevents degenerate solutions.

### 3D fusion
The pipeline avoids explicit R/t calibration between tele and wide by fusing:
- **Depth** — from the tele image (large angular size → high accuracy)
- **Direction** — from the wide image via a local homography H

This works under the assumption that the two cameras share approximately the
same optical centre (common in zoom-lens or beam-splitter setups).

---

## License

MIT
