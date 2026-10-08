# Plan: Object recognition with SIFT, ratio test and RANSAC homography

> You are responsible for managing computer assets in a busy computing lab. Implement a computer vision solution using the Scale-Invariant Feature Transform (SIFT) to automatically recognize and identify individual computer systems and their unique components, such as monitors and keyboards, to maintain an up-to-date inventory.

## What the task needs

- **Goal:** find the reference object in the scene and outline it
- **Input:** a reference image and a scene image
- **Method:** Object recognition with SIFT, ratio test and RANSAC homography (chosen by cue words: 'sift', 'recogniz', 'assets')
- **Also considered:** Screen detection and status with Hough lines (3), Geometric transformation (1), Texture and shape features (HOG and LBP) (1)
- **Limits to state in your answer:** Flat or repetitive surfaces give weak/ambiguous matches. descriptors can be None. The Python constructor is cv.BFMatcher, not cv.BruteForceMatcher. Guard empty descriptors and short pairs. Four matches suffice algebraically but not necessarily for reliability. Collinear/repetitive points and parallax can fool a model.

## Steps at a glance

1. Load the reference image and the scene image
2. Detect SIFT keypoints and descriptors in both images
3. Match descriptors with Lowe's ratio test
4. Estimate the homography with RANSAC
5. Draw the recognised object outline
6. Show every stage side by side and save the result

## Step 1: Load the reference image and the scene image

imread returns None (no exception) for a wrong path, so check it before using the image.

```python
ref = cv.imread('reference.jpg')
img = cv.imread('scene.jpg')
if ref is None or img is None:
    raise FileNotFoundError('check the image paths')
```

**Watch out:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths.

*Source: Lab 01 Manual p.14*

## Step 2: Detect SIFT keypoints and descriptors in both images

SIFT finds blob-like keypoints over a scale space (difference of Gaussians) and describes each with a 128-value gradient histogram, so the same point matches across scale and rotation. Descriptors can be None for flat images.

```python
sift = cv.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(cv.cvtColor(ref, cv.COLOR_BGR2GRAY), None)
kp, des = sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)
report['keypoints_ref'] = len(kp_ref)
report['keypoints_scene'] = len(kp)
```

**Watch out:** cv.SIFT_create needs OpenCV 4.4 or newer (the patent expired); older builds put it in xfeatures2d.

**From the course:** SIFT step 1, scale-space extrema detection, finds keypoint candidates at multiple scales using a Difference of Gaussians (DoG). *(Lab Manual 06 p.18)*

**From the course:** The SIFT code creates the detector with cv2.SIFT_create() and calls sift.detectAndCompute(gray, None), returning keypoints and descriptors. (Note: descriptors is None when no keypoints are found (correction C28).) *(Lab Manual 06 p.23)*

*Source: Lab Manual 06 p.18; Lab Manual 06 p.20; Lab Manual 06 p.23*

## Step 3: Match descriptors with Lowe's ratio test

For each reference descriptor take its two nearest scene descriptors (L2 distance for SIFT). Keep the match only if the best is clearly better than the second best (ratio < 0.75); ambiguous matches are dropped.

Parameters:
- `RATIO = 0.75`: Lowe's ratio; lower = stricter

```python
good = []
if des_ref is not None and des is not None and len(des) >= 2:
    for pair in cv.BFMatcher(cv.NORM_L2).knnMatch(des_ref, des, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            good.append(pair[0])
report['good_matches'] = len(good)
```

**Watch out:** knnMatch can return fewer than 2 neighbours for some descriptors; check len(pair) == 2.

**From the course:** SIFT keypoint matching compares descriptors using Euclidean distance or other similarity measures. *(Lab Manual 06 p.21)*

*Source: Lab Manual 06 p.21; Lab 06 Tasks p.1*

## Step 4: Estimate the homography with RANSAC

A homography needs at least 4 point pairs. RANSAC fits it to random 4-pair samples and keeps the model most pairs agree with (error below RANSAC_PX), so wrong matches are ignored. Too few inliers = object not present.

Parameters:
- `RANSAC_PX = 5.0`: reprojection error allowed, pixels
- `MIN_INLIERS = 10`: inliers needed to accept

```python
H, inliers = None, 0
if len(good) >= 4:
    src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv.findHomography(src, dst, cv.RANSAC, RANSAC_PX)
    inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
report['inliers'] = inliers
report['found'] = H is not None and inliers >= MIN_INLIERS
```

**Watch out:** findHomography needs at least 4 matches and returns None when it fails.

**From the course:** After outlier rejection, the remaining matches estimate a homography between images, used for image stitching or object recognition. *(Lab Manual 06 p.22)*

**From the course:** RANSAC keeps only geometrically consistent matches (inliers). *(Lab Manual 06 p.22)*

*Source: Lab Manual 06 p.21; Lab Manual 06 p.22; Lab 03 Manual p.18; Lab 03 Tasks p.1*

## Step 5: Draw the recognised object outline

The reference image's four corners are mapped through H into the scene; the polygon is the object's outline, which also shows its rotation and perspective.

```python
vis = img.copy()
if report['found']:
    h, w = ref.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    box = cv.perspectiveTransform(corners, H)
    cv.polylines(vis, [np.int32(box)], True, (0, 255, 0), 3)
    report['box'] = np.int32(box).reshape(-1, 2).tolist()
cv.putText(vis, 'FOUND' if report['found'] else 'NOT FOUND', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
           (0, 255, 0) if report['found'] else (0, 0, 255), 2)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center. Flat or repetitive surfaces give weak/ambiguous matches. descriptors can be None.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19; Lab Manual 06 p.18; Lab Manual 06 p.20*

## Step 6: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: SIFT asset inventory (L06-T02)

The knowledge base has a tested solution for a similar lab task (match 0.21). Its steps:

1. Provide named reference pictures for assets.
2. Compute SIFT descriptors and match with BF L2 ratio test.
3. Validate planar localization by RANSAC.
4. Return evidence for each reference and explicit failure reasons.

Limits: Identical-looking assets cannot be uniquely identified by appearance alone. Multiple instances and untextured screens require extensions.

Code (`solutions/lab06.py`, `lab06.task02_inventory`; helpers come from `solutions/cv_core.py`):

```python
def task02_inventory(scene,references):
    """references maps asset label->reference image. Similar-looking items may be ambiguous."""
    return {name:sift_match(ref,scene) for name,ref in references.items()}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Object recognition with SIFT, ratio test and RANSAC homography.

Task: You are responsible for managing computer assets in a busy computing lab. Implement a computer
      vision solution using the Scale-Invariant Feature Transform (SIFT) to automatically recognize
      and identify individual computer systems and their unique components, such as monitors and
      keyboards, to maintain an up-to-date inventory.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
RATIO = 0.75               # Lowe's ratio; lower = stricter
RANSAC_PX = 5.0            # reprojection error allowed, pixels
MIN_INLIERS = 10           # inliers needed to accept


def run(ref, img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Detect SIFT keypoints and descriptors in both images
    sift = cv.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(cv.cvtColor(ref, cv.COLOR_BGR2GRAY), None)
    kp, des = sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)
    report['keypoints_ref'] = len(kp_ref)
    report['keypoints_scene'] = len(kp)

    # Step 3: Match descriptors with Lowe's ratio test
    good = []
    if des_ref is not None and des is not None and len(des) >= 2:
        for pair in cv.BFMatcher(cv.NORM_L2).knnMatch(des_ref, des, k=2):
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                good.append(pair[0])
    report['good_matches'] = len(good)

    # Step 4: Estimate the homography with RANSAC
    H, inliers = None, 0
    if len(good) >= 4:
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inlier_mask = cv.findHomography(src, dst, cv.RANSAC, RANSAC_PX)
        inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
    report['inliers'] = inliers
    report['found'] = H is not None and inliers >= MIN_INLIERS

    # Step 5: Draw the recognised object outline
    vis = img.copy()
    if report['found']:
        h, w = ref.shape[:2]
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        box = cv.perspectiveTransform(corners, H)
        cv.polylines(vis, [np.int32(box)], True, (0, 255, 0), 3)
        report['box'] = np.int32(box).reshape(-1, 2).tolist()
    cv.putText(vis, 'FOUND' if report['found'] else 'NOT FOUND', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
               (0, 255, 0) if report['found'] else (0, 0, 255), 2)
    stages['Result'] = vis
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
    ref = cv.imread(sys.argv[1] if len(sys.argv) > 1 else 'reference.jpg')
    img = cv.imread(sys.argv[2] if len(sys.argv) > 2 else 'scene.jpg')
    if ref is None or img is None:
        raise FileNotFoundError('check the reference and scene paths')
    stages, report = run(ref, img)
    for name, value in report.items():
        print(f'{name}: {value}')
    show(img, stages)
    cv.imwrite('result.png', stages['Result'])

if __name__ == '__main__':
    main()
```
