# Plan: Basic image operations

> Threshold the image and then rotate it 45 degrees with scaling.

## What the task needs

- **Goal:** carry out the requested basic operations
- **Input:** one image
- **Method:** Basic image operations (chosen by cue words: 'thresh_simple', 'rotate')
- **Also considered:** Thresholding: global vs Otsu vs adaptive (3), Geometric transformation (3)
- **Limits to state in your answer:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Threshold the image
4. Rotate (and scale) about the center
5. Show every stage side by side and save the result

## Step 1: Load the image and check it

imread returns None (no exception) for a wrong path, so check it before using the image. Color images load as BGR.

```python
img = cv.imread('input.jpg')
if img is None:
    raise FileNotFoundError('input.jpg')
```

**Watch out:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths.

*Source: Lab 01 Manual p.14*

## Step 2: Convert to grayscale

Edge, line, circle and threshold operations work on one intensity channel. OpenCV loads color as BGR, so the code must be COLOR_BGR2GRAY (not RGB2GRAY).

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
stages['Grayscale'] = gray
```

**Watch out:** cv.imread gives BGR order; using COLOR_RGB2GRAY swaps the red and blue weights.

**From the course:** cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) converts a color image to grayscale. *(Lab 01 Manual p.15)*

*Source: Lab 01 Manual p.15; Lab 05 Manual p.9*

## Step 3: Threshold the image

cv.threshold returns (threshold used, image); pixels above THRESHOLD become 255, the rest 0.

Parameters:
- `THRESHOLD = 127`: cutoff 0-255

```python
_, mask = cv.threshold(gray, THRESHOLD, 255, cv.THRESH_BINARY)
img = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)       # later steps work on the thresholded image
stages['Threshold'] = mask
```

**Watch out:** maxval=200 means intensity200, not255. A plotting program may rescale it to white.

**From the course:** Otsu in OpenCV: cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); the given threshold 0 is ignored. *(Lab 05 Manual p.9)*

**From the course:** cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY): 100 is the threshold value, 200 is the maximum value assigned to pixels above it. (Note: The manual calls 200 pure white; in uint8 pure white is 255 (correction C05).) *(Lab 01 Manual p.21)*

*Source: Lab 01 Manual p.21; Lab 05 Manual p.8*

## Step 4: Rotate (and scale) about the center

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

## Step 5: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: Thresholding and 45-degree scaled rotation (L01-T07)

The knowledge base has a tested solution for a similar lab task (match 0.32). Its steps:

1. Convert to grayscale.
2. Compare global 127 with local Gaussian threshold.
3. Build getRotationMatrix2D at 45 degrees and scale 0.8.
4. Expand output bounds to preserve transformed content.

Limits: Scale 0.8 alone does not prevent cropping on a same-size square canvas; expansion corrects that assumption.

Code (`solutions/lab01.py`, `lab01.task07_threshold_rotate`; helpers come from `solutions/cv_core.py`):

```python
def task07_threshold_rotate(image):
    g=gray(image); h,w=g.shape
    _,fixed=cv.threshold(g,127,255,cv.THRESH_BINARY)
    adaptive=cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,7)
    M=cv.getRotationMatrix2D(((w-1)/2,(h-1)/2),45,0.8)
    H=np.vstack([M,[0,0,1]])
    rotated,_=warp_fit(image,H)
    return {'Grayscale':g,'Global 127':fixed,'Adaptive':adaptive,'45deg scale0.8 expanded':rotated}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Basic image operations.

Task: Threshold the image and then rotate it 45 degrees with scaling.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
THRESHOLD = 127            # cutoff 0-255
ANGLE = 45                 # degrees, positive = counter-clockwise
SCALE = 1.0                # scale factor


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Threshold the image
    _, mask = cv.threshold(gray, THRESHOLD, 255, cv.THRESH_BINARY)
    img = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)       # later steps work on the thresholded image
    stages['Threshold'] = mask

    # Step 4: Rotate (and scale) about the center
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
