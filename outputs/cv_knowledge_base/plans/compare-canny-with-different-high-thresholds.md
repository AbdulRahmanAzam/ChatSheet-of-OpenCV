# Plan: Edge detection with Canny

> Compare Canny with different high thresholds.

## What the task needs

- **Goal:** find the object edges and compare thresholds
- **Input:** one image
- **Method:** Edge detection with Canny (chosen by cue words: 'canny')
- **Also considered:** Basic image operations (2)
- **Limits to state in your answer:** Use explicit blur to control smoothing. Non-maximum suppression is not simply a weak-pixel threshold. Sobel-Feldman is another name for Sobel, not a separate diagonal detector. A derivative response is not yet a thin binary edge map.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Smooth noise with a Gaussian blur
4. Detect edges with Canny
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

## Step 4: Detect edges with Canny

Canny gives thin, connected edges (gradient, non-maximum suppression, hysteresis). Pixels above CANNY_HIGH are sure edges; pixels between the two thresholds survive only if connected to a sure edge. A 1:2 to 1:3 ratio is the usual choice. Raise both if texture gives too many edges.

Parameters:
- `CANNY_LOW = 50`: hysteresis low threshold
- `CANNY_HIGH = 150`: hysteresis high threshold

```python
edges = cv.Canny(smooth, CANNY_LOW, CANNY_HIGH)
stages['Canny edges'] = edges
```

**Watch out:** Too low thresholds give texture edges that later vote for false lines/circles; too high break real boundaries.

**From the course:** Hysteresis thresholding (Canny) uses two thresholds: a high threshold to detect strong edges and a low threshold to link weak edges; example cv2.Canny(image, 100, 200). *(Lab 05 Manual p.10)*

**From the course:** Canny edge detection steps: Gaussian smoothing reduces noise, gradient calculation finds magnitude and direction, non-maximum suppression, and hysteresis thresholding. *(Lab Manual 06 p.11)*

*Source: Lab 04 Manual p.22; Lab Manual 06 p.11; Lab Manual 06 p.13*

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

## Closest worked lab task: Canny high-threshold comparison (L05-T05)

The knowledge base has a tested solution for a similar lab task (match 0.39). Its steps:

1. Convert and Gaussian smooth.
2. Hold low threshold 30 constant and try high 60,120,200.
3. Compare connected weak edges and suppression.

Limits: Final output cannot identify original strong/weak classes; that would require access to intermediate gradients/NMS.

Code (`solutions/lab05.py`, `lab05.task05_canny`; helpers come from `solutions/cv_core.py`):

```python
def task05_canny(image):
    g=cv.GaussianBlur(gray(image),(5,5),1.2)
    out={'Smoothed':g}
    for low,high in [(30,60),(30,120),(30,200)]:
        out[f'Canny low={low}, high={high}']=cv.Canny(g,low,high,L2gradient=True)
    return out
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Edge detection with Canny.

Task: Compare Canny with different high thresholds.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 5             # odd Gaussian kernel size; raise to 7-9 for noisy images
CANNY_LOW = 50             # hysteresis low threshold
CANNY_HIGH = 150           # hysteresis high threshold


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Smooth noise with a Gaussian blur
    smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    stages['Blurred'] = smooth

    # Step 4: Detect edges with Canny
    edges = cv.Canny(smooth, CANNY_LOW, CANNY_HIGH)
    stages['Canny edges'] = edges
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
