# Plan: Geometric transformation

> Undo the rotation of a satellite image using sin and cos.

## What the task needs

- **Goal:** apply or undo the geometric transformation
- **Input:** one image
- **Method:** Geometric transformation (chosen by cue words: 'rotat')
- **Also considered:** Basic image operations (2)
- **Limits to state in your answer:** Affine does not preserve every length or angle. Three collinear points are degenerate. Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography.

## Steps at a glance

1. Load the image and check it
2. Rotate (and scale) about the center
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

## Step 2: Rotate (and scale) about the center

getRotationMatrix2D builds the 2x3 matrix for rotation about a center (positive angle = counter-clockwise in the displayed image) with optional scale; warpAffine applies it.

Parameters:
- `ANGLE = 45`: degrees, positive = counter-clockwise
- `SCALE = 1.0`: scale factor

```python
h, w = img.shape[:2]
M = cv.getRotationMatrix2D((w / 2, h / 2), ANGLE, SCALE)
warped = cv.warpAffine(img, M, (w, h))
report['matrix'] = np.round(M, 3).tolist()
stages[f'Rotated {ANGLE} deg'] = warped
```

**Watch out:** warpAffine keeps the original size, so rotated corners are cut off unless the output size is enlarged.

**From the course:** cv2.getRotationMatrix2D(center, angle, scale) computes a 2x3 rotation matrix; positive angles rotate counter-clockwise. *(Lab 01 Manual p.22)*

**From the course:** OpenCV's warpAffine() accepts a 2x3 matrix while warpPerspective() uses a 3x3 matrix. *(Lab 03 Manual p.8)*

*Source: Lab 03 Manual p.5; Lab 03 Manual p.11; Lab 03 Manual p.13; Lab 03 Tasks p.2*

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

## Closest worked lab task: Undo satellite rotation with sin/cos (L03-T02)

The knowledge base has a tested solution for a similar lab task (match 0.4). Its steps:

1. Build a 2x2 inverse rotation from sine/cosine.
2. Center the rotation using homogeneous translations.
3. Transform corners, compute bounds, and warp to an expanded canvas.

Limits: Default corrects a 45 degree clockwise image rotation. Direction must match actual input.

Code (`solutions/lab03.py`, `lab03.task02_undo_rotation`; helpers come from `solutions/cv_core.py`):

```python
def task02_undo_rotation(image,observed_clockwise_degrees=45):
    h,w=image.shape[:2]; cx,cy=(w-1)/2,(h-1)/2
    H=translation(cx,cy)@rotation(-observed_clockwise_degrees)@translation(-cx,-cy)
    return warp_fit(image,H)
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Geometric transformation.

Task: Undo the rotation of a satellite image using sin and cos.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
ANGLE = 45                 # degrees, positive = counter-clockwise
SCALE = 1.0                # scale factor


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Rotate (and scale) about the center
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), ANGLE, SCALE)
    warped = cv.warpAffine(img, M, (w, h))
    report['matrix'] = np.round(M, 3).tolist()
    stages[f'Rotated {ANGLE} deg'] = warped
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
