# Plan: Marker-based watershed segmentation

> Implement the complete marker-based watershed pipeline.

## What the task needs

- **Goal:** split touching objects into separate regions and count them
- **Input:** one image
- **Method:** Marker-based watershed segmentation (chosen by cue words: 'watershed', 'marker')
- **Limits to state in your answer:** cv.watershed takes a uint8 3-channel image and int32 markers. Unknown starts at0, boundaries become-1.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Smooth noise with a Gaussian blur
4. Otsu automatic threshold
5. Separate touching objects with marker-based watershed
6. Draw watershed boundaries
7. Show every stage side by side and save the result

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

## Step 3: Smooth noise with a Gaussian blur

Noise creates false edges and false Hough votes. A Gaussian blur removes fine noise while keeping strong boundaries. Kernel size must be odd; bigger = smoother but weaker thin edges.

Parameters:
- `BLUR_KSIZE = 5`: odd Gaussian kernel size; raise to 7-9 for noisy images

```python
smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
stages['Blurred'] = smooth
```

**Watch out:** Too much blur merges nearby edges (two screens, two lane dashes); too little lets noise vote.

**From the course:** cv2.GaussianBlur(image, (31, 31), 0) blurs with a 31x31 kernel; kernel sizes must be odd numbers. (Note: The code comment mentions 5x5 but the code uses 31x31 (correction C03).) *(Lab 01 Manual p.17)*

**From the course:** Gaussian blur uses a Gaussian-shaped kernel and gives smoother results than box blur; often used for noise reduction. *(Lab 04 Manual p.17)*

*Source: Lab 01 Manual p.17; Lab 04 Manual p.17*

## Step 4: Otsu automatic threshold

Otsu picks the cutoff that best separates two intensity groups (minimum within-class variance), so it suits a two-peak (bimodal) histogram. The threshold argument 0 is ignored. Blurring first makes the peaks cleaner.

Parameters:
- `THRESH_TYPE = cv.THRESH_BINARY`: or cv.THRESH_BINARY_INV for dark objects

```python
otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
report['otsu_threshold'] = float(otsu_value)
stages[f'Otsu (T={otsu_value:.0f})'] = mask
```

**Watch out:** Otsu assumes two peaks in the histogram; with uneven lighting use adaptive thresholding.

**From the course:** Otsu in OpenCV: cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); the given threshold 0 is ignored. *(Lab 05 Manual p.9)*

**From the course:** Otsu's thresholding automatically selects an optimal threshold that maximizes the inter-class variance between object and background pixels. *(Lab 05 Manual p.9)*

*Source: Lab 05 Manual p.9; Lab 05 Tasks p.3*

## Step 5: Separate touching objects with marker-based watershed

Touching objects form one blob after thresholding. The distance transform peaks at each object center; thresholding it (DIST_FRACTION of the max) leaves one seed per object. Seeds are labelled, the unknown band is set to 0, and watershed floods from the seeds; boundaries come back as -1. Lower DIST_FRACTION if two objects share one seed, raise it if one object gets two. (The Lab 05 manual code uses 0.2 * max; that keeps touching objects joined, so 0.5 is used here to split them.)

Parameters:
- `DIST_FRACTION = 0.5`: fraction of max distance for sure foreground

```python
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv.dilate(opening, kernel, iterations=3)
dist = cv.distanceTransform(opening, cv.DIST_L2, 5)
_, sure_fg = cv.threshold(dist, DIST_FRACTION * dist.max(), 255, cv.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
_, markers = cv.connectedComponents(sure_fg)
markers = markers + 1          # background becomes 1, not 0
markers[unknown == 255] = 0    # 0 = let watershed decide
markers = cv.watershed(img, markers)
report['count'] = len(set(np.unique(markers)) - {-1, 1})
stages['Distance transform'] = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
stages['Sure foreground'] = sure_fg
```

**Watch out:** cv.watershed needs a 3-channel image and int32 markers; it changes the markers array in place.

**From the course:** The watershed code gets the sure foreground by thresholding cv2.distanceTransform(opening, cv2.DIST_L2, 5) at 0.2 * max. *(Lab 05 Manual p.13)*

**From the course:** The unknown region is sure background minus sure foreground (cv2.subtract); markers come from cv2.connectedComponents, +1 so background is 1, and unknown set to 0. *(Lab 05 Manual p.13)*

*Source: Lab 05 Manual p.12; Lab 05 Manual p.13; Lab 05 Tasks p.7; Lab 05 Tasks p.8*

## Step 6: Draw watershed boundaries

Pixels labelled -1 are the watershed lines between regions.

```python
vis = img.copy()
vis[markers == -1] = (0, 0, 255)
cv.putText(vis, f"Objects: {report['count']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
stages['Result'] = vis
```

**Watch out:** cv.watershed takes a uint8 3-channel image and int32 markers. Unknown starts at0, boundaries become-1.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

**From the course:** After cv2.watershed(image, markers), boundary pixels are labeled -1 and are colored to show the segmentation. (Note: In BGR, red is [0, 0, 255]; the manual uses [255, 0, 0] (correction C23).) *(Lab 05 Manual p.13)*

*Source: Lab 05 Manual p.12; Lab 05 Manual p.13; Lab 05 Tasks p.7*

## Step 7: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: Complete marker-based watershed pipeline (L05-T07)

The knowledge base has a tested solution for a similar lab task (match 0.36). Its steps:

1. Convert to grayscale and choose foreground polarity.
2. Otsu threshold; open the mask.
3. Dilate to background support.
4. Distance transform and threshold for foreground.
5. Subtract to find unknown.
6. Label seeds; add 1 so known background=1; unknown=0.
7. Run watershed on BGR uint8 with int32 markers.
8. Mark boundaries red and show all stages.

Limits: Red is BGR(0,0,255). Empty/poor markers can leave objects unseparated.

Code (`solutions/lab05.py`, `lab05.task07_watershed`; helpers come from `solutions/cv_core.py`):

```python
def task07_watershed(image,foreground='bright'):
    return watershed_stages(image,0.5,foreground)
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Marker-based watershed segmentation.

Task: Implement the complete marker-based watershed pipeline.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 5             # odd Gaussian kernel size; raise to 7-9 for noisy images
THRESH_TYPE = cv.THRESH_BINARY # or cv.THRESH_BINARY_INV for dark objects
DIST_FRACTION = 0.5        # fraction of max distance for sure foreground


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Smooth noise with a Gaussian blur
    smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    stages['Blurred'] = smooth

    # Step 4: Otsu automatic threshold
    otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
    report['otsu_threshold'] = float(otsu_value)
    stages[f'Otsu (T={otsu_value:.0f})'] = mask

    # Step 5: Separate touching objects with marker-based watershed
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, DIST_FRACTION * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1          # background becomes 1, not 0
    markers[unknown == 255] = 0    # 0 = let watershed decide
    markers = cv.watershed(img, markers)
    report['count'] = len(set(np.unique(markers)) - {-1, 1})
    stages['Distance transform'] = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    stages['Sure foreground'] = sure_fg

    # Step 6: Draw watershed boundaries
    vis = img.copy()
    vis[markers == -1] = (0, 0, 255)
    cv.putText(vis, f"Objects: {report['count']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
