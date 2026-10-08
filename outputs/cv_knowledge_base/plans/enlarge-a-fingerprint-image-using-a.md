# Plan: Geometric transformation

> Enlarge a fingerprint image using a manual scaling matrix.

## What the task needs

- **Goal:** apply or undo the geometric transformation
- **Input:** one image
- **Method:** Geometric transformation (chosen by cue words: 'enlarg')
- **Limits to state in your answer:** Affine does not preserve every length or angle. Three collinear points are degenerate. Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography.

## Steps at a glance

1. Load the image and check it
2. Scale with a scaling matrix
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

## Step 2: Scale with a scaling matrix

Scaling matrix [[sx,0,0],[0,sy,0]]: x is multiplied by sx and y by sy. The output size must be enlarged by the same factors or the result is cut off. Cubic interpolation fills the new pixels smoothly.

Parameters:
- `SX = 2.0`: horizontal scale
- `SY = 2.0`: vertical scale

```python
h, w = img.shape[:2]
M = np.float32([[SX, 0, 0], [0, SY, 0]])
warped = cv.warpAffine(img, M, (int(w * SX), int(h * SY)), flags=cv.INTER_CUBIC)
report['matrix'] = M.tolist()
stages['Scaled'] = warped
```

**Watch out:** Affine does not preserve every length or angle. Three collinear points are degenerate. dsize is (width,height). Resizing two medical images to the same size does not register their anatomy.

**From the course:** cv2.warpAffine(image, matrix, (width, height)) applies the transformation; with the same canvas size rotated corners are cut off and empty space is black. *(Lab 01 Manual p.22)*

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

## Closest worked lab task: Manual matrix fingerprint enlargement (L03-T01)

The knowledge base has a tested solution for a similar lab task (match 0.23). Its steps:

1. Construct a diagonal 2D scale matrix and embed in 3x3 homogeneous form.
2. Translate source center to origin and destination center into place.
3. Warp to three-times width and height.

Limits: Task says enlarge by 300 percent: default interprets as to 300 percent (factor 3); use 4 for a literal 300 percent increase.

Code (`solutions/lab03.py`, `lab03.task01_scale`; helpers come from `solutions/cv_core.py`):

```python
def task01_scale(image,factor=3.0):
    """Default means 'to 300%'. For literal 'increase by 300%' use factor=4."""
    if factor<=0: raise ValueError('factor must be positive')
    h,w=image.shape[:2]; target=(round(w*factor),round(h*factor))
    S=np.diag([factor,factor,1.])
    H=translation((target[0]-1)/2,(target[1]-1)/2)@S@translation(-(w-1)/2,-(h-1)/2)
    return cv.warpPerspective(image,H,target),H
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Geometric transformation.

Task: Enlarge a fingerprint image using a manual scaling matrix.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
SX = 2.0                   # horizontal scale
SY = 2.0                   # vertical scale


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Scale with a scaling matrix
    h, w = img.shape[:2]
    M = np.float32([[SX, 0, 0], [0, SY, 0]])
    warped = cv.warpAffine(img, M, (int(w * SX), int(h * SY)), flags=cv.INTER_CUBIC)
    report['matrix'] = M.tolist()
    stages['Scaled'] = warped
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
