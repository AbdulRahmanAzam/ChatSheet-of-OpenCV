# Plan: Geometric transformation

> Rectify a square seen in perspective into a top-down view.

## What the task needs

- **Goal:** apply or undo the geometric transformation
- **Input:** one image
- **Method:** Geometric transformation (chosen by cue words: 'rectif')
- **Limits to state in your answer:** Affine does not preserve every length or angle. Three collinear points are degenerate. Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography.

## Steps at a glance

1. Load the image and check it
2. Perspective (top-down) rectification from four corners
3. Show every stage side by side and save the result

## Step 1: Load the image and check it

imread returns None (no exception) for a wrong path, so check it before using the image. Color images load as BGR.

```python
img = cv.imread('input.jpg')
if img is None:
    raise FileNotFoundError('input.jpg')
```

**Watch out:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths.

*Source: Lab 01 Manual p.14*

## Step 2: Perspective (top-down) rectification from four corners

A homography has 8 unknowns, so 4 point pairs fix it. Give the corners in the same order as the output rectangle (top-left, top-right, bottom-right, bottom-left).

Parameters:
- `SRC_POINTS = [[60, 40], [340, 70], [360, 330], [40, 300]]`: four corners TL, TR, BR, BL
- `OUT_W = 400`: output width
- `OUT_H = 400`: output height

```python
src = np.float32(SRC_POINTS)
dst = np.float32([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]])
P = cv.getPerspectiveTransform(src, dst)
warped = cv.warpPerspective(img, P, (OUT_W, OUT_H))
report['matrix'] = np.round(P, 4).tolist()
stages['Top-down view'] = warped
```

**Watch out:** The four source points must be in the same order as the destination corners, or the image flips.

**From the course:** OpenCV's warpAffine() accepts a 2x3 matrix while warpPerspective() uses a 3x3 matrix. *(Lab 03 Manual p.8)*

**From the course:** A homography is a transformation that relates two images of the same planar surface. *(Lab 01 Manual p.5)*

*Source: Lab 03 Manual p.18; Lab 03 Tasks p.1; Lab 03 Tasks p.2*

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

## Closest worked lab task: Top-down square perspective rectification (L03-T08)

The knowledge base has a tested solution for a similar lab task (match 0.4). Its steps:

1. Provide four ordered convex corners.
2. Associate them with square corners.
3. Estimate getPerspectiveTransform and warpPerspective.

Limits: Corner order TL,TR,BR,BL; target 400x400 default may distort a nonsquare physical object.

Code (`solutions/lab03.py`, `lab03.task08_rectify`; helpers come from `solutions/cv_core.py`):

```python
def task08_rectify(image,corners,size=(400,400)):
    return perspective_quad(image,corners,size)
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Geometric transformation.

Task: Rectify a square seen in perspective into a top-down view.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
SRC_POINTS = [[60, 40], [340, 70], [360, 330], [40, 300]] # four corners TL, TR, BR, BL
OUT_W = 400                # output width
OUT_H = 400                # output height


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Perspective (top-down) rectification from four corners
    src = np.float32(SRC_POINTS)
    dst = np.float32([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]])
    P = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, P, (OUT_W, OUT_H))
    report['matrix'] = np.round(P, 4).tolist()
    stages['Top-down view'] = warped
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
    path = sys.argv[1] if len(sys.argv) > 1 else 'input.jpg'
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    stages, report = run(img)
    for name, value in report.items():
        print(f'{name}: {value}' if not isinstance(value, np.ndarray) else f'{name}: array {value.shape}')
    show(img, stages)
    if 'Result' in stages:
        cv.imwrite('result.png', stages['Result'])

if __name__ == '__main__':
    main()
```
