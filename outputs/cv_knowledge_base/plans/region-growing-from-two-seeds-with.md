# Plan: Region growing from seeds

> Region growing from two seeds with three tolerances.

## What the task needs

- **Goal:** grow regions from seed points and compare tolerances
- **Input:** one image
- **Method:** Region growing from seeds (chosen by cue words: 'region grow', 'seeds')
- **Limits to state in your answer:** Seed order is (x,y) here; the manual uses array-style coordinates. Convert intensity values to int before subtraction.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Region growing from seed points
4. Show every stage side by side and save the result

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

## Step 3: Region growing from seed points

Region growing starts at a seed and adds 4-connected neighbours whose intensity is within the tolerance of the seed (FIXED_RANGE compares to the seed, not to the neighbour). Small tolerance stops early; large tolerance leaks into the background.

Parameters:
- `SEEDS = [(50, 50)]`: seed points (x, y) inside the regions you want
- `TOLERANCES = [5, 15, 30]`: intensity tolerances to compare

```python
h, w = gray.shape
mask = np.zeros((h, w), np.uint8)
for tol in TOLERANCES:
    region = np.zeros((h, w), np.uint8)
    for sx, sy in SEEDS:
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv.floodFill(gray.copy(), ff, (sx, sy), 255, tol, tol,
                     4 | cv.FLOODFILL_MASK_ONLY | cv.FLOODFILL_FIXED_RANGE | (255 << 8))
        region |= ff[1:-1, 1:-1]
    stages[f'Region, tolerance {tol}'] = region
    report[f'area_tol{tol}'] = int((region > 0).sum())
    mask = region
```

**Watch out:** floodFill needs a mask 2 pixels larger than the image in both directions.

**From the course:** Region growing starts with a seed pixel and expands to neighboring pixels that are similar by a criterion such as intensity or color. *(Lab 05 Manual p.10)*

**From the course:** In region growing, a pixel whose intensity difference from the seed is below the threshold is added to the segmented region (mask set to 255) and its neighbours are pushed on the stack. *(Lab 05 Manual p.11)*

*Source: Lab 05 Manual p.10; Lab 05 Manual p.11*

## Step 4: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: Two seeds and three region-growth tolerances (L05-T06)

The knowledge base has a tested solution for a similar lab task (match 0.56). Its steps:

1. Provide two seed points.
2. Grow with fixed-seed tolerances 5,15,30.
3. Compare six masks for leakage and incomplete coverage.

Limits: Course MRI and seeds absent. This is intensity segmentation, not a clinical lesion classifier.

Code (`solutions/lab05.py`, `lab05.task06_region_grow`; helpers come from `solutions/cv_core.py`):

```python
def task06_region_grow(image,seeds,tolerances=(5,15,30)):
    if len(seeds)!=2: raise ValueError('Supply two (x,y) seeds')
    return {f'seed={seed}, tolerance={t}':region_grow(image,seed,t) for seed in seeds for t in tolerances}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Region growing from seeds.

Task: Region growing from two seeds with three tolerances.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
SEEDS = [(50, 50)]         # seed points (x, y) inside the regions you want
TOLERANCES = [5, 15, 30]   # intensity tolerances to compare


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Region growing from seed points
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    for tol in TOLERANCES:
        region = np.zeros((h, w), np.uint8)
        for sx, sy in SEEDS:
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv.floodFill(gray.copy(), ff, (sx, sy), 255, tol, tol,
                         4 | cv.FLOODFILL_MASK_ONLY | cv.FLOODFILL_FIXED_RANGE | (255 << 8))
            region |= ff[1:-1, 1:-1]
        stages[f'Region, tolerance {tol}'] = region
        report[f'area_tol{tol}'] = int((region > 0).sum())
        mask = region
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
