# Wide-Tele 3D Ball Localisation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A vision-based pipeline for **high-precision 3D localisation of survey markers** on slopes,
using a single zoom camera operating in two focal lengths — wide for scene-level detection,
tele for sub-pixel measurement. No stereo rig, no LiDAR, no baseline calibration required.

![Pipeline](demo_output/00_pipeline_diagram.jpg)

---

## Application

Long-term displacement monitoring of slope surfaces requires repeated, precise measurement
of fixed reference targets across a wide spatial range. Traditional survey instruments
(total stations, GNSS) are accurate but require on-site access; photogrammetric methods
need multi-view setups or dense stereo rigs.

This pipeline achieves centimetre-level 3D localisation of **orange spherical markers**
from a single fixed camera position, automating what would otherwise require manual
measurement or expensive instrumentation.

---

## Why a zoom camera — and why it creates a hard constraint

The system uses a **varifocal (zoom) lens**, capturing both a wide-angle overview and a
telephoto close-up from the **same optical centre**. This design is intentional: it
mirrors how biological vision systems work — a wide field of view for scene-level
detection, a narrow field for precise measurement — and avoids the mechanical complexity
and calibration burden of a two-camera rig.

But it introduces a fundamental geometric constraint:

> Because the optical centre does not move between focal lengths, **there is no baseline**
> between the wide and tele images. Standard stereo triangulation — which recovers depth
> from the disparity between two viewpoints — is impossible.

The tele→wide pixel mapping is a pure rotation + zoom, described entirely by a 3×3
homography H. There is no translation component, no epipolar geometry, no disparity.

**Depth must therefore come from a different source: the known physical size of the
marker.** A sphere of radius R that subtends angular radius α satisfies:

```
D = R / sin(α)
```

This equation is the geometric foundation of the entire pipeline. Every algorithmic
choice — subpixel boundary detection, multi-scale registration, distortion correction —
exists to make `α` as accurate as possible.

---

## Pipeline

Three stages. The wide image locates targets; the tele image measures them precisely;
the two are fused to recover 3D position.

---

### Stage 1 — Subpixel marker detection (tele image)

The telephoto image provides high angular resolution on each marker. Because `α` is
computed directly from the fitted circle radius, circle-fit error propagates linearly
into distance error — making subpixel accuracy here a direct requirement for
centimetre-level 3D output.

**Coarse detection:** HSV segmentation in both red-hue wrap regions (0° and 180°),
followed by morphological cleaning and circularity filtering, gives an initial bounding
circle for each candidate.

**Red-likelihood map:** rather than using the binary mask for edge detection, a smooth
per-pixel redness score `L = hue_closeness × saturation × value^γ` provides a continuous
boundary signal. This avoids hard thresholding artefacts near shadow boundaries and
specular highlights.

![Likelihood map](demo_output/03_likelihood_map.jpg)

**Subpixel boundary sampling:** radial profiles are cast at 720 angles, sampled at
0.5 px steps. Each edge is located via gradient-peak parabola fit, then refined by
cross-threshold interpolation. An alpha-sweep over threshold blend levels selects the
value that minimises the subsequent fitting residual.

**IRLS circle fit:** the ~700 accepted boundary points are fitted with Iteratively
Re-weighted Least Squares (Huber M-estimator):

```
e_i = ||p_i − c|| − r
w_i = min(1,  k·σ / |e_i|)     k = 1.345,   σ = 1.4826 · MAD(e)
```

Points corrupted by specular highlights or partial occlusion are down-weighted, not
discarded. The covariance matrix yields per-parameter uncertainty.

![Ball #0 fit](demo_output/tele_target_00_overlay.jpg)

| | cx | cy | r | Coverage |
|---|---|---|---|---|
| Marker #0 | 2673.45 px | 2226.74 px | 352.80 px | 97.9% |
| Std | ±0.086 px | ±0.092 px | ±0.063 px | — |

![Ball #1 fit](demo_output/tele_target_01_overlay.jpg)

| | cx | cy | r | Coverage |
|---|---|---|---|---|
| Marker #1 | 4961.96 px | 1350.80 px | 280.59 px | 98.3% |
| Std | ±0.185 px | ±0.191 px | ±0.133 px | — |

Sub-pixel centre accuracy **< 0.2 px std**, 98% angular coverage on both markers.

---

### Stage 2 — Wide-tele image registration

To convert tele pixel coordinates into wide camera rays, the homography H\_total that
maps tele pixels to wide pixels must be estimated. Because the scale ratio between
wide and tele is **unknown at runtime** (it changes with zoom position), this cannot
be pre-calibrated and must be solved per-image.

Registration is **target-driven**: the search is anchored to the detected marker centre
rather than the full image, concentrating computation where accuracy matters and
avoiding sensitivity to background clutter at very different scales.

**Multi-scale template matching:** 33 candidate scale values are searched using
gradient-magnitude features. Each candidate is scored as a log-posterior:

```
logpost = NCC  +  w_PSR · PSR  +  w_prior · log p(s)
```

PSR (Peak-to-Sidelobe Ratio) measures peak sharpness — high PSR indicates an
unambiguous match. A log-Gaussian prior on scale prevents degenerate solutions.

**ECC refinement:** the best scale is refined to sub-pixel accuracy by maximising
the Enhanced Correlation Coefficient, giving a full 2×3 Euclidean warp.

The final homography composes all three layers:

```
H_total = T · ECC · S      (scale × ECC warp × template-match translation)

H_total =
⎡ 0.20625  −0.00117  466.77 ⎤
⎢ 0.00117   0.20625  1448.85 ⎥
⎣ 0.00000   0.00000    1.00  ⎦
```

The 0.206 diagonal confirms the zoom ratio. The 0.33° implied rotation is consistent
with a zoom lens introducing negligible rotation between focal lengths.

![Registration blend](demo_output/04_registration_blend.jpg)

*Alpha blend of tele (warped) onto wide. The green crosshair marks the tele marker
centre projected via H\_total — it lands on the wide marker.*

| ECC ρ | Photometric RMSE | Scale |
|-------|-----------------|-------|
| 0.8013 | 0.1521 | 0.206 |

---

### Stage 3 — 3D fusion

With a subpixel circle fit from Stage 1 and H\_total from Stage 2, depth and direction
are estimated independently and combined:

```
α  =  median  atan2(||r_c × r_b||,  r_c · r_b)   over all ~700 boundary rays
D  =  R / sin(α)
P  =  D · undistort( H_total · p_tele )
```

**Depth** comes from the tele image: the marker subtends a large angle (~0.3°), making
the angular-radius estimate highly sensitive to small distance changes.

**Direction** comes from the wide camera ray: the wide image provides a stable
absolute pointing reference with minimal distortion sensitivity.

Using the **median of all boundary ray angles** — rather than the single fitted
radius — provides additional robustness to any outlier boundary points retained by
IRLS.

![3D result](demo_output/05_3d_result.jpg)

| Marker | D (m) | X (m) | Y (m) | Z (m) | α (°) |
|--------|-------|-------|-------|-------|-------|
| #0 | 17.129 | −2.430 | +0.143 | 16.955 | 0.3345 |
| #1 | 21.571 | −2.244 | −0.132 | 21.454 | 0.2656 |

Tele optical axis relative to wide: **yaw = −8.83°, pitch = +0.34°**

**Dual-branch consistency check:** as an internal validation, the same depth D is
independently fused with the tele ray rotated via a yaw/pitch matrix derived from
H\_total — a completely different geometric path. Agreement between the two confirms
that registration and depth estimation are mutually consistent.

| Marker | ‖P\_H − P\_R‖ |
|--------|--------------|
| #0 | **3.9 mm** |
| #1 | **6.9 mm** |

---

## Run the demo

```bash
pip install opencv-python numpy
python ball_3d_localization_demo.py
```

All three modules run on synthetic images — no dataset required. To reproduce the
field figures above:

```bash
python generate_demo_visuals.py   # requires Img238.jpg (wide) + Img333.jpg (tele)
```

---

## Repository structure

```
ball_3d_localization_demo.py    Core algorithm modules + synthetic smoke test
generate_demo_visuals.py        Full pipeline on real field images → demo_output/
demo_output/                    All output figures
```

---

## Dependencies

`opencv-python ≥ 4.5`  ·  `numpy ≥ 1.22`  ·  `Python ≥ 3.10`

---

## License

MIT
