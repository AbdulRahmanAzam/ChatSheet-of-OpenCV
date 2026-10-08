# Plan: Texture and shape features (HOG and LBP)

> Analyse material texture using HOG and LBP.

## What the task needs

- **Goal:** describe the texture/shape with feature vectors
- **Input:** one image
- **Method:** Texture and shape features (HOG and LBP) (chosen by cue words: 'hog', 'material')
- **Limits to state in your answer:** A descriptor alone does not classify an object. Window, cell, block, stride and bin choices must match training. Noise changes threshold comparisons. Rotation properties depend on variant. The manual contrast code is a local sum, not the stated standard deviation, and its uint8 square can overflow. Other fields use different energy definitions such as squared filter responses: name the definition.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Compute Sobel gradients, magnitude and direction
4. HOG descriptor (shape / edge-direction features)
5. LBP histogram (texture features)
6. Show every stage side by side and save the result

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

## Step 3: Compute Sobel gradients, magnitude and direction

Sobel in x and y with a float depth (CV_64F) keeps negative slopes; magnitude = sqrt(gx^2+gy^2) and direction = atan2(gy, gx).

```python
gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gx ** 2 + gy ** 2)
direction = np.degrees(np.arctan2(gy, gx))
stages['Gradient magnitude'] = cv.convertScaleAbs(magnitude)
```

**Watch out:** Sobel-Feldman is another name for Sobel, not a separate diagonal detector. A derivative response is not yet a thin binary edge map. Do not compute orientation or zero crossings from the absolute uint8 image.

**From the course:** Sobel total gradient magnitude G = sqrt(Gx^2 + Gy^2) combines horizontal and vertical changes. *(Lab Manual 06 p.8)*

**From the course:** The HED code computes gradient_x and gradient_y with cv2.Sobel(..., cv2.CV_64F, ...) and magnitude np.sqrt(gradient_x**2 + gradient_y**2). *(Lab 04 Manual p.11)*

*Source: Lab 04 Manual p.20; Lab 04 Manual p.21; Lab Manual 06 p.6; Lab Manual 06 p.15*

## Step 4: HOG descriptor (shape / edge-direction features)

HOG divides the window into 8x8 cells, builds a 9-bin gradient-direction histogram per cell and normalizes over 2x2-cell blocks; for 64x128 that is 7*15*36 = 3780 values. It describes shape and structure.

```python
patch = cv.resize(gray, (64, 128))
hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
hog_vec = hog.compute(patch).ravel()
report['hog_length'] = len(hog_vec)
```

**Watch out:** A descriptor alone does not classify an object. Window, cell, block, stride and bin choices must match training.

**From the course:** HOG block normalization groups neighboring cells into blocks (typical block 2x2 cells) and normalizes them to reduce the effects of lighting variations. *(Lab 04 Manual p.7)*

**From the course:** HOG captures information about local gradient or edge patterns, the distribution of gradient directions in an image. *(Lab 04 Manual p.6)*

*Source: Lab 04 Manual p.6; Lab 04 Manual p.7*

## Step 5: LBP histogram (texture features)

LBP compares each pixel with its 8 neighbours (1 if neighbour >= center) to make an 8-bit code; the histogram of codes describes texture and does not change with uniform brightness shifts.

```python
lbp = lbp_image(gray)
lbp_hist = np.bincount(lbp.ravel(), minlength=256).astype(float)
lbp_hist /= lbp_hist.sum()
stages['LBP'] = lbp
report['lbp_bins'] = len(lbp_hist)

# helper used above (defined once, outside run):
def lbp_image(g):
    """8-neighbour, radius-1 local binary pattern codes (border pixels skipped)."""
    g = g.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(shifts):
        n = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((n >= c).astype(np.uint8) << bit)
    return code
```

**Watch out:** Noise changes threshold comparisons. Rotation properties depend on variant. The manual contrast code is a local sum, not the stated standard deviation, and its uint8 square can overflow. Other fields use different energy definitions such as squared filter responses: name the definition.

**From the course:** In LBP, if the neighbor intensity is greater than or equal to the center pixel intensity, the neighbor is assigned 1. *(Lab 04 Manual p.8)*

**From the course:** In the LBP code radius = 1 is the radius of the circular neighborhood and n_points = 8 * radius is the number of neighboring pixels to consider. *(Lab 04 Manual p.8)*

*Source: Lab 04 Manual p.7; Lab 04 Manual p.8; Lab 04 Manual p.9; Lab 04 Manual p.13*

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

## Closest worked lab task: Material texture analysis with HOG and LBP (L04-T01)

The knowledge base has a tested solution for a similar lab task (match 0.48). Its steps:

1. Compute circular uniform LBP histogram for local microtexture.
2. Compute normalized block HOG for directional structure.
3. Compare optional local energy and contrast.
4. For an actual label, compare with labeled exemplars via classify_material.

Limits: LBP: useful microtexture, weak under noise/scale change. HOG: useful grain/weave direction, weak under rotation and texture ambiguity. No wood/metal/fabric accuracy claimed without a labeled dataset.

Code (`solutions/lab04.py`, `lab04.task01_features`; helpers come from `solutions/cv_core.py`):

```python
def task01_features(image):
    labels,lbp=uniform_lbp(image); energy,contrast=texture_stats(image)
    return {'lbp_image':labels,'lbp_histogram':lbp,'hog':hog_features(image),
            'energy':energy,'local_contrast':contrast}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Texture and shape features (HOG and LBP).

Task: Analyse material texture using HOG and LBP.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----



def lbp_image(g):
    """8-neighbour, radius-1 local binary pattern codes (border pixels skipped)."""
    g = g.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(shifts):
        n = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((n >= c).astype(np.uint8) << bit)
    return code


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Compute Sobel gradients, magnitude and direction
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.degrees(np.arctan2(gy, gx))
    stages['Gradient magnitude'] = cv.convertScaleAbs(magnitude)

    # Step 4: HOG descriptor (shape / edge-direction features)
    patch = cv.resize(gray, (64, 128))
    hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_vec = hog.compute(patch).ravel()
    report['hog_length'] = len(hog_vec)

    # Step 5: LBP histogram (texture features)
    lbp = lbp_image(gray)
    lbp_hist = np.bincount(lbp.ravel(), minlength=256).astype(float)
    lbp_hist /= lbp_hist.sum()
    stages['LBP'] = lbp
    report['lbp_bins'] = len(lbp_hist)
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
