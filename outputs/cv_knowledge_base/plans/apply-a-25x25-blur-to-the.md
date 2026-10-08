# Plan: Basic image operations

> Apply a 25x25 blur to the image and extract the exact center region of interest.

## What the task needs

- **Goal:** carry out the requested basic operations
- **Input:** one image
- **Method:** Basic image operations (chosen by cue words: 'blur_box', 'center_roi')
- **Also considered:** Color segmentation in HSV (1)
- **Limits to state in your answer:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1.

## Steps at a glance

1. Load the image and check it
2. Blur the image with a large kernel
3. Crop the exact center region (ROI)
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

## Step 2: Blur the image with a large kernel

cv.blur averages each KxK neighbourhood (box filter); a 25x25 kernel gives a strong blur.

Parameters:
- `BOX_KSIZE = 25`: kernel size

```python
blurred = cv.blur(img, (BOX_KSIZE, BOX_KSIZE))
stages['Blurred'] = blurred
```

**Watch out:** An unnormalized box filter returns a sum, which can overflow or saturate. It is not a local standard-deviation calculation.

**From the course:** Box blur replaces each pixel with the average of its neighboring pixels within a square kernel; useful for noise reduction and smoothing. *(Lab 04 Manual p.16)*

**From the course:** The box blur kernel in the manual is np.ones((3,3), dtype=np.float32) / 9 applied with cv2.filter2D(image, -1, kernel). *(Lab 04 Manual p.16)*

*Source: Lab 04 Manual p.16*

## Step 3: Crop the exact center region (ROI)

NumPy slicing is [rows, cols] = [y, x]; the start is (size - roi) // 2 on each axis.

Parameters:
- `ROI_W = 200`: width
- `ROI_H = 200`: height

```python
h, w = img.shape[:2]
y0, x0 = (h - ROI_H) // 2, (w - ROI_W) // 2
roi = img[y0:y0 + ROI_H, x0:x0 + ROI_W]
stages['Center ROI'] = roi
```

**Watch out:** A slice may be a view: modifying it can modify the original image. Clamping coordinates silently can change the requested crop size. Image y increases downward. A standard Cartesian positive-angle matrix therefore appears clockwise on an image.

**From the course:** The first column vector of a 2x2 transformation matrix is the new X-axis (image of the unit vector (1,0)); the second column is the new Y-axis (image of (0,1)). *(Lab 03 Manual p.10)*

**From the course:** In matrix [[a, b], [c, d]], c (Y of the new X-axis) gives vertical shearing and b (X of the new Y-axis) gives horizontal shearing. *(Lab 03 Manual p.10)*

*Source: Lab 01 Manual p.18; Lab 01 Tasks p.2; Lab 03 Manual p.15; Lab 03 Manual p.17*

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

## Closest worked lab task: 25x25 blur and exact center ROI (L01-T05)

The knowledge base has a tested solution for a similar lab task (match 0.39). Its steps:

1. Check image is at least 300x300.
2. Blur the whole original with a 25x25 Gaussian.
3. Extract the same center 300x300 coordinates from both.
4. Plot side by side with titles and axes hidden.

Limits: High-resolution course image missing; compare actual texture retention once supplied.

Code (`solutions/lab01.py`, `lab01.task05_blur_roi`; helpers come from `solutions/cv_core.py`):

```python
def task05_blur_roi(image):
    h,w=image.shape[:2]
    if min(h,w)<300: raise ValueError('Image must be at least 300x300')
    blurred=cv.GaussianBlur(image,(25,25),0)
    x,y=(w-300)//2,(h-300)//2
    return {'Original center 300x300':image[y:y+300,x:x+300],
            'Blurred center 300x300':blurred[y:y+300,x:x+300]}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Basic image operations.

Task: Apply a 25x25 blur to the image and extract the exact center region of interest.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BOX_KSIZE = 25             # kernel size
ROI_W = 200                # width
ROI_H = 200                # height


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Blur the image with a large kernel
    blurred = cv.blur(img, (BOX_KSIZE, BOX_KSIZE))
    stages['Blurred'] = blurred

    # Step 3: Crop the exact center region (ROI)
    h, w = img.shape[:2]
    y0, x0 = (h - ROI_H) // 2, (w - ROI_W) // 2
    roi = img[y0:y0 + ROI_H, x0:x0 + ROI_W]
    stages['Center ROI'] = roi
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
