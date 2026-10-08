# Plan: Color clustering with k-means

> Segment an image with k-means for K=2,4,6.

## What the task needs

- **Goal:** group pixels into K color clusters and compare K values
- **Input:** one image
- **Method:** Color clustering with k-means (chosen by cue words: 'k-means', 'k=')
- **Limits to state in your answer:** Color clusters are not guaranteed semantic objects or connected regions. K controls model size, and initialization can affect results.

## Steps at a glance

1. Load the image and check it
2. Cluster pixel colors with k-means
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

## Step 2: Cluster pixel colors with k-means

k-means groups pixels into K color clusters and repaints each pixel with its cluster center. Data must be float32 of shape (N,3). Small K merges regions, large K splits them. Results depend on initialization, so attempts=3 with k-means++ is used.

Parameters:
- `K_VALUES = [2, 4, 6]`: cluster counts to try

```python
pixels = img.reshape(-1, 3).astype(np.float32)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
segmented = {}
for k in K_VALUES:
    _, labels, centers = cv.kmeans(pixels, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
    segmented[k] = np.uint8(centers)[labels.flatten()].reshape(img.shape)
    stages[f'K = {k}'] = segmented[k]
    report[f'colors_k{k}'] = len(np.unique(labels))
```

**Watch out:** cv.kmeans needs float32 data shaped (N, 3) and returns labels shaped (N, 1); flatten before indexing.

**From the course:** Clustering-based segmentation groups pixels into clusters by feature similarity using K-Means, Mean-Shift or DBSCAN. *(Lab 05 Manual p.14)*

**From the course:** Clustering techniques like K-Means or Mean-Shift group similar pixels by feature similarity such as color values. *(Lab 05 Manual p.5)*

*Source: Lab 05 Manual p.14; Lab 05 Manual p.15; Lab 05 Tasks p.9*

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

## Closest worked lab task: K=2,4,6 image clustering (L05-T09)

The knowledge base has a tested solution for a similar lab task (match 0.5). Its steps:

1. Reshape BGR pixels to float32 rows.
2. Run K-means for 2,4,6 with recorded seed and stopping criteria.
3. Reconstruct center-colored images.
4. Compare compactness, color detail and false merges.

Limits: Spatial position is not included in this baseline; disconnected areas of the same color share a label.

Code (`solutions/lab05.py`, `lab05.task09_kmeans`; helpers come from `solutions/cv_core.py`):

```python
def task09_kmeans(image):
    result={}
    for k in (2,4,6):
        segmented,labels,compactness=segment_kmeans(image,k)
        result[k]={'image':segmented,'labels':labels,'compactness':compactness}
    return result
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Color clustering with k-means.

Task: Segment an image with k-means for K=2,4,6.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
K_VALUES = [2, 4, 6]       # cluster counts to try


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Cluster pixel colors with k-means
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    segmented = {}
    for k in K_VALUES:
        _, labels, centers = cv.kmeans(pixels, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
        segmented[k] = np.uint8(centers)[labels.flatten()].reshape(img.shape)
        stages[f'K = {k}'] = segmented[k]
        report[f'colors_k{k}'] = len(np.unique(labels))
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
