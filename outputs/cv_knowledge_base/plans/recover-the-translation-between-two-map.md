# Plan: Geometric transformation

> Recover the translation between two map images and shift it back.

## What the task needs

- **Goal:** apply or undo the geometric transformation
- **Input:** one image
- **Method:** Geometric transformation (chosen by cue words: 'translat')
- **Limits to state in your answer:** Affine does not preserve every length or angle. Three collinear points are degenerate. Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography.

## Steps at a glance

1. Load the image and check it
2. Translate (shift) the image
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

## Step 2: Translate (shift) the image

Translation matrix [[1,0,tx],[0,1,ty]]: x moves by tx, y by ty (down is positive).

Parameters:
- `TX = 50`: shift right, pixels
- `TY = 30`: shift down, pixels

```python
h, w = img.shape[:2]
M = np.float32([[1, 0, TX], [0, 1, TY]])
warped = cv.warpAffine(img, M, (w, h))
stages['Translated'] = warped
```

**Watch out:** Affine does not preserve every length or angle. Three collinear points are degenerate.

**From the course:** cv2.warpAffine(image, matrix, (width, height)) applies the transformation; with the same canvas size rotated corners are cut off and empty space is black. *(Lab 01 Manual p.22)*

**From the course:** Horizontal shear adds a fraction k of one coordinate to another: x' = x + ky, with matrix [[1, k], [0, 1]]. *(Lab 03 Manual p.15)*

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

## Closest worked lab task: Map translation recovery (L03-T04)

The knowledge base has a tested solution for a similar lab task (match 0.22). Its steps:

1. Undo 150 pixels left and 80 pixels up using translation(+150,+80).
2. Create an output canvas that accommodates the positive shift.
3. Warp with the 3x3 matrix.

Limits: Translation cannot restore image content that was already cropped away in the input.

Code (`solutions/lab03.py`, `lab03.task04_translate`; helpers come from `solutions/cv_core.py`):

```python
def task04_translate(image,offset=(150,80),canvas=None):
    H=translation(*offset); h,w=image.shape[:2]
    if canvas is None: canvas=(w+max(0,int(offset[0])),h+max(0,int(offset[1])))
    return cv.warpPerspective(image,H,canvas),H
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Geometric transformation.

Task: Recover the translation between two map images and shift it back.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
TX = 50                    # shift right, pixels
TY = 30                    # shift down, pixels


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Translate (shift) the image
    h, w = img.shape[:2]
    M = np.float32([[1, 0, TX], [0, 1, TY]])
    warped = cv.warpAffine(img, M, (w, h))
    stages['Translated'] = warped
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
