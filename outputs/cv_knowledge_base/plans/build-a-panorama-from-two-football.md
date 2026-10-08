# Plan: Panorama stitching with SIFT and homography

> Build a panorama from two football field images using point correspondences.

## What the task needs

- **Goal:** align overlapping images and stitch them into one panorama
- **Input:** several overlapping images
- **Method:** Panorama stitching with SIFT and homography (chosen by cue words: 'panoram')
- **Limits to state in your answer:** Black pixels can be valid image content, so intensity>0 is not a validity mask. Sequential stitching accumulates drift; parallax, exposure differences and moving objects need more advanced handling.

## Steps at a glance

1. Load the overlapping images in left-to-right order
2. Warp and blend the images into a panorama
3. Show every stage side by side and save the result

## Step 1: Load the overlapping images in left-to-right order

Neighbouring images must overlap (about 30% or more) or there is nothing to match.

```python
imgs = [cv.imread(p) for p in paths]
if any(i is None for i in imgs) or len(imgs) < 2:
    raise FileNotFoundError('need at least two readable images')
```

**Watch out:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths.

*Source: Lab 01 Manual p.14*

## Step 2: Warp and blend the images into a panorama

For each new image: SIFT keypoints, ratio-test matches, RANSAC homography that maps it onto the current panorama, then warp it onto a canvas big enough for both. The canvas is shifted so nothing gets cut off.

Parameters:
- `RATIO = 0.75`: Lowe's ratio

```python
pano = imgs[0]
for nxt in imgs[1:]:
    sift = cv.SIFT_create()
    k1, d1 = sift.detectAndCompute(cv.cvtColor(nxt, cv.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv.cvtColor(pano, cv.COLOR_BGR2GRAY), None)
    pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < 4:
        raise ValueError('Not enough overlap to stitch')
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)        # maps nxt -> pano
    pano = warp_pair(pano, nxt, H)
    report.setdefault('matches_per_pair', []).append(len(good))
vis = pano
report['panorama_size'] = vis.shape[1::-1]
stages['Panorama'] = vis

# helper used above (defined once, outside run):
def warp_pair(base, new, H):
    """Warp new into base's frame with H (new -> base) on a canvas that fits both."""
    hb, wb = base.shape[:2]; hn, wn = new.shape[:2]
    corners = np.float32([[0, 0], [wn, 0], [wn, hn], [0, hn]]).reshape(-1, 1, 2)
    moved = cv.perspectiveTransform(corners, H)
    allc = np.concatenate([moved, np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)])
    xmin, ymin = np.floor(allc.min(axis=(0, 1))).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=(0, 1))).astype(int)
    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], float)
    canvas = cv.warpPerspective(new, shift @ H, (xmax - xmin, ymax - ymin))
    region = canvas[-ymin:-ymin + hb, -xmin:-xmin + wb]
    keep = base.sum(axis=2) > 0
    region[keep] = base[keep]
    return canvas
```

**Watch out:** Images must overlap and be ordered; a wrong homography makes the warped image explode in size.

**From the course:** cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) converts a color image to grayscale. *(Lab 01 Manual p.15)*

**From the course:** The SIFT code creates the detector with cv2.SIFT_create() and calls sift.detectAndCompute(gray, None), returning keypoints and descriptors. (Note: descriptors is None when no keypoints are found (correction C28).) *(Lab Manual 06 p.23)*

*Source: Lab 03 Tasks p.1; Lab 06 Tasks p.2; Lab Manual 06 p.18; Lab Manual 06 p.20*

## Step 3: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: Panorama from field correspondences (L03-T09)

The knowledge base has a tested solution for a similar lab task (match 0.19). Its steps:

1. Provide at least four paired landmarks.
2. Estimate right-to-left homography and reject failure.
3. Compute full canvas bounds.
4. Warp validity masks and average overlapping valid content.

Limits: Planar scene or approximately pure camera rotation assumed; parallax and exposure seams remain limitations.

Code (`solutions/lab03.py`, `lab03.task09_panorama`; helpers come from `solutions/cv_core.py`):

```python
def task09_panorama(left,right,left_points,right_points):
    a,b=np.asarray(left_points,np.float32),np.asarray(right_points,np.float32)
    if a.shape != b.shape or a.ndim!=2 or a.shape[1]!=2 or len(a)<4:
        raise ValueError('At least four paired points required')
    H,inliers=cv.findHomography(b,a,cv.RANSAC,3.0)
    if H is None or inliers is None or inliers.sum()<4: raise ValueError('Degenerate correspondences')
    result,T=stitch_with_homography(left,right,H)
    return result,H,T
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Panorama stitching with SIFT and homography.

Task: Build a panorama from two football field images using point correspondences.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
RATIO = 0.75               # Lowe's ratio


def warp_pair(base, new, H):
    """Warp new into base's frame with H (new -> base) on a canvas that fits both."""
    hb, wb = base.shape[:2]; hn, wn = new.shape[:2]
    corners = np.float32([[0, 0], [wn, 0], [wn, hn], [0, hn]]).reshape(-1, 1, 2)
    moved = cv.perspectiveTransform(corners, H)
    allc = np.concatenate([moved, np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)])
    xmin, ymin = np.floor(allc.min(axis=(0, 1))).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=(0, 1))).astype(int)
    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], float)
    canvas = cv.warpPerspective(new, shift @ H, (xmax - xmin, ymax - ymin))
    region = canvas[-ymin:-ymin + hb, -xmin:-xmin + wb]
    keep = base.sum(axis=2) > 0
    region[keep] = base[keep]
    return canvas


def run(imgs):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Warp and blend the images into a panorama
    pano = imgs[0]
    for nxt in imgs[1:]:
        sift = cv.SIFT_create()
        k1, d1 = sift.detectAndCompute(cv.cvtColor(nxt, cv.COLOR_BGR2GRAY), None)
        k2, d2 = sift.detectAndCompute(cv.cvtColor(pano, cv.COLOR_BGR2GRAY), None)
        pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(d1, d2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < 4:
            raise ValueError('Not enough overlap to stitch')
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)        # maps nxt -> pano
        pano = warp_pair(pano, nxt, H)
        report.setdefault('matches_per_pair', []).append(len(good))
    vis = pano
    report['panorama_size'] = vis.shape[1::-1]
    stages['Panorama'] = vis
    return stages, report


def show(original, stages):
    """Original plus every stage in one matplotlib figure (BGR converted to RGB)."""
    items = [('Original', original)] + [(k, v) for k, v in stages.items() if isinstance(v, np.ndarray) and v.ndim in (2, 3)]
    cols = min(3, len(items)); rows = (len(items) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, image) in enumerate(items, 1):
        plt.subplot(rows, cols, i)
        if image.ndim == 3:
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(name); plt.axis('off')
    plt.tight_layout(); plt.show()

def main():
    paths = sys.argv[1:] or ['left.jpg', 'right.jpg']
    imgs = [cv.imread(p) for p in paths]
    if len(imgs) < 2 or any(i is None for i in imgs):
        raise FileNotFoundError('need at least two readable images')
    stages, report = run(imgs)
    for name, value in report.items():
        print(f'{name}: {value}')
    show(imgs[0], stages)
    cv.imwrite('panorama.png', stages['Panorama'])

if __name__ == '__main__':
    main()
```
