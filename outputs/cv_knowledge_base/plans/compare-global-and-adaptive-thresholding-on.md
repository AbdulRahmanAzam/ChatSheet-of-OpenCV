# Plan: Thresholding: global vs Otsu vs adaptive

> Compare global and adaptive thresholding on a document image.

## What the task needs

- **Goal:** turn the image into foreground/background and compare threshold methods
- **Input:** one image
- **Method:** Thresholding: global vs Otsu vs adaptive (chosen by cue words: 'adaptive', 'document')
- **Also considered:** Basic image operations (2)
- **Limits to state in your answer:** maxval=200 means intensity200, not255. A plotting program may rescale it to white. blockSize must be odd and greater than1, with uint8 single-channel input. Increasing C lowers the threshold: more white pixels for BINARY and fewer for BINARY_INV. Otsu does not fix spatially varying illumination or guarantee semantic objects. Blur/contrast transforms change the histogram and can change the selected threshold.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Smooth noise with a Gaussian blur
4. Global threshold
5. Otsu automatic threshold
6. Adaptive (local) threshold
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

## Step 4: Global threshold

One cutoff for the whole image: fine when lighting is even. THRESH_BINARY_INV makes dark objects white, which contour and morphology functions expect.

Parameters:
- `THRESHOLD = 127`: fixed cutoff 0-255
- `THRESH_TYPE = cv.THRESH_BINARY_INV`: or cv.THRESH_BINARY_INV for dark objects

```python
_, mask = cv.threshold(smooth, THRESHOLD, 255, THRESH_TYPE)
stages['Global threshold'] = mask
```

**Watch out:** One cutoff fails under uneven lighting; compare with Otsu and adaptive.

**From the course:** The global threshold example is cv2.threshold(image, 128, 255, cv2.THRESH_BINARY). *(Lab 05 Manual p.8)*

**From the course:** Global thresholding applies a single threshold value to the entire image; effective when objects and background have a clear intensity difference. *(Lab 05 Manual p.8)*

*Source: Lab 01 Manual p.21; Lab 05 Manual p.8*

## Step 5: Otsu automatic threshold

Otsu picks the cutoff that best separates two intensity groups (minimum within-class variance), so it suits a two-peak (bimodal) histogram. The threshold argument 0 is ignored. Blurring first makes the peaks cleaner.

Parameters:
- `THRESH_TYPE = cv.THRESH_BINARY_INV`: or cv.THRESH_BINARY_INV for dark objects

```python
otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
report['otsu_threshold'] = float(otsu_value)
stages[f'Otsu (T={otsu_value:.0f})'] = mask
```

**Watch out:** Otsu assumes two peaks in the histogram; with uneven lighting use adaptive thresholding.

**From the course:** Otsu in OpenCV: cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); the given threshold 0 is ignored. *(Lab 05 Manual p.9)*

**From the course:** Otsu's thresholding automatically selects an optimal threshold that maximizes the inter-class variance between object and background pixels. *(Lab 05 Manual p.9)*

*Source: Lab 05 Manual p.9; Lab 05 Tasks p.3*

## Step 6: Adaptive (local) threshold

Each pixel gets its own cutoff from its neighbourhood (weighted mean minus C), so shadows and uneven light do not ruin it. BLOCK_SIZE must be odd and bigger than the strokes/objects; C shifts the cutoff.

Parameters:
- `BLOCK_SIZE = 31`: odd neighbourhood size
- `C = 10`: constant subtracted from the local mean
- `THRESH_TYPE = cv.THRESH_BINARY_INV`: or cv.THRESH_BINARY_INV for dark objects

```python
mask = cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TYPE, BLOCK_SIZE, C)
stages['Adaptive threshold'] = mask
```

**Watch out:** BLOCK_SIZE must be odd and > 1; a block smaller than the strokes hollows out thick text.

**From the course:** The adaptive example is cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2): block size 11 and constant C = 2. *(Lab 05 Manual p.9)*

**From the course:** Adaptive thresholding uses different threshold values for different regions, adapting to local intensity variations; ideal for varying lighting conditions. *(Lab 05 Manual p.8)*

*Source: Lab 05 Manual p.8; Lab 05 Manual p.9; Lab 05 Tasks p.2*

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

## Closest worked lab task: Global versus adaptive document segmentation (L05-T01)

The knowledge base has a tested solution for a similar lab task (match 0.24). Its steps:

1. Apply global inverse thresholds 80,127,180.
2. Apply adaptive Gaussian inverse threshold.
3. Compare masks and illumination failures; with known truth use task10_compare.

Limits: Threshold polarity is dark text as foreground. Which method is best depends on the supplied document.

Code (`solutions/lab05.py`, `lab05.task01_document`; helpers come from `solutions/cv_core.py`):

```python
def task01_document(image):
    g=gray(image); out={'Original':g}
    for t in (80,127,180): out[f'Global {t}']=cv.threshold(g,t,255,cv.THRESH_BINARY_INV)[1]
    out['Adaptive Gaussian']=cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,31,7)
    return out
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Thresholding: global vs Otsu vs adaptive.

Task: Compare global and adaptive thresholding on a document image.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 5             # odd Gaussian kernel size; raise to 7-9 for noisy images
THRESHOLD = 127            # fixed cutoff 0-255
THRESH_TYPE = cv.THRESH_BINARY_INV # or cv.THRESH_BINARY_INV for dark objects
BLOCK_SIZE = 31            # odd neighbourhood size
C = 10                     # constant subtracted from the local mean


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Smooth noise with a Gaussian blur
    smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    stages['Blurred'] = smooth

    # Step 4: Global threshold
    _, mask = cv.threshold(smooth, THRESHOLD, 255, THRESH_TYPE)
    stages['Global threshold'] = mask

    # Step 5: Otsu automatic threshold
    otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
    report['otsu_threshold'] = float(otsu_value)
    stages[f'Otsu (T={otsu_value:.0f})'] = mask

    # Step 6: Adaptive (local) threshold
    mask = cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TYPE, BLOCK_SIZE, C)
    stages['Adaptive threshold'] = mask
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
