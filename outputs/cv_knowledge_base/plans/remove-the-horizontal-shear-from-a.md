# Plan: Geometric transformation

> Remove the horizontal shear from a barcode image.

## What the task needs

- **Goal:** apply or undo the geometric transformation
- **Input:** one image
- **Method:** Geometric transformation (chosen by cue words: 'shear')
- **Limits to state in your answer:** Affine does not preserve every length or angle. Three collinear points are degenerate. Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography.

## Steps at a glance

1. Load the image and check it
2. Shear (or undo a shear)
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

## Step 2: Shear (or undo a shear)

Horizontal shear x' = x + sh*y slants vertical lines; use the negative value to undo a known shear.

Parameters:
- `SHEAR_X = -0.3`: horizontal shear factor

```python
h, w = img.shape[:2]
M = np.float32([[1, SHEAR_X, 0], [0, 1, 0]])
warped = cv.warpAffine(img, M, (int(w + abs(SHEAR_X) * h), h))
stages['Sheared'] = warped
```

**Watch out:** Affine does not preserve every length or angle. Three collinear points are degenerate.

**From the course:** Horizontal shear adds a fraction k of one coordinate to another: x' = x + ky, with matrix [[1, k], [0, 1]]. *(Lab 03 Manual p.15)*

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

## Closest worked lab task: Barcode horizontal de-shearing (L03-T03)

The knowledge base has a tested solution for a similar lab task (match 0.36). Its steps:

1. Represent observed horizontal shear as xprime=x+k*y.
2. Use inverse coefficient minus k.
3. Warp to expanded output bounds.

Limits: Shear coefficient is not supplied by task; default 0.3 is illustrative and configurable.

Code (`solutions/lab03.py`, `lab03.task03_deshear`; helpers come from `solutions/cv_core.py`):

```python
def task03_deshear(image,observed_shear=0.3):
    H=np.array([[1.,-observed_shear,0],[0,1,0],[0,0,1]])
    return warp_fit(image,H)
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Geometric transformation.

Task: Remove the horizontal shear from a barcode image.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
SHEAR_X = -0.3             # horizontal shear factor


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Shear (or undo a shear)
    h, w = img.shape[:2]
    M = np.float32([[1, SHEAR_X, 0], [0, 1, 0]])
    warped = cv.warpAffine(img, M, (int(w + abs(SHEAR_X) * h), h))
    stages['Sheared'] = warped
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
