# Plan: Contrast enhancement and pseudocolor

> Enhance an X-ray image so the bones are clearer and apply pseudocolor.

## What the task needs

- **Goal:** make the important detail visible and measure the improvement
- **Input:** one image
- **Method:** Contrast enhancement and pseudocolor (chosen by cue words: 'enhanc', 'pseudocolor', 'x-ray')
- **Limits to state in your answer:** equalizeHist expects uint8 single-channel input. Independently equalizing BGR channels alters colors; work on a luminance channel when color fidelity matters. Some libraries define a reciprocal gamma parameter; state the convention. Convert before arithmetic.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Local contrast enhancement (CLAHE)
4. Pseudocolor for visual inspection
5. Measure the contrast gain
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

## Step 3: Local contrast enhancement (CLAHE)

CLAHE equalizes each tile separately and clips the histogram, so it brings out local detail (bones, tissue, text) without blowing out noise the way global equalization can.

Parameters:
- `CLAHE_CLIP = 2.0`: contrast limit; higher = stronger, noisier
- `CLAHE_TILES = 8`: tiles per side

```python
clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
enhanced = clahe.apply(gray)
stages['CLAHE'] = enhanced
```

**Watch out:** CLAHE works on one channel; for color, apply it to the L channel of LAB.

**From the course:** cv2.equalizeHist spreads clumped pixel intensities across the full 0-255 range, improving global contrast of dark or washed-out images. *(Lab 01 Manual p.24)*

*Source: Lab 01 Manual p.24; Lab 02 Tasks p.4*

## Step 4: Pseudocolor for visual inspection

The eye separates hues better than gray levels; a colormap makes intensity differences easy to see. It is a display aid, not new information.

Parameters:
- `COLORMAP = cv.COLORMAP_JET`: any cv.COLORMAP_* constant

```python
colored = cv.applyColorMap(enhanced, COLORMAP)
stages['Pseudocolor'] = colored
```

**Watch out:** JET colors in an echocardiogram are not Doppler measurements. Keep the original scalar image available and avoid interpreting chosen colors as clinical labels.

*Source: Lab 02 Tasks p.4; Lab 02 Tasks p.6*

## Step 5: Measure the contrast gain

The standard deviation of intensity is a simple contrast number; it should go up after enhancement.

```python
report['contrast_before'] = round(float(gray.std()), 2)
report['contrast_after'] = round(float(enhanced.std()), 2)
```

**Watch out:** Specify range and normalization consistently. Different images can have the same histogram.

*Source: Lab 04 Manual p.9; Lab 05 Tasks p.3*

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

## Closest worked lab task: X-ray enhancement and pseudocolor (L02-T01)

The knowledge base has a tested solution for a similar lab task (match 0.3). Its steps:

1. Convert to grayscale and equalize histogram.
2. Apply JET colormap.
3. Apply explicitly simulated channel gains.
4. Threshold high intensities.
5. Compare log and gamma 0.6 transforms.

Limits: 8-bit educational visualization; no lesion diagnosis or physical density calibration. Original images absent.

Code (`solutions/lab02.py`, `lab02.task01_xray`; helpers come from `solutions/cv_core.py`):

```python
def task01_xray(image, dense_threshold=180, gamma=0.6, bgr_gains=(0.9,1.0,1.1)):
    g=gray(image); equalized=cv.equalizeHist(g)
    heat=cv.applyColorMap(equalized,cv.COLORMAP_JET)
    balance=u8(heat.astype(float)*np.asarray(bgr_gains))
    _,dense=cv.threshold(equalized,dense_threshold,255,cv.THRESH_BINARY)
    return {'Original gray':g,'Equalized':equalized,'JET intensity':heat,
            'Simulated channel gains':balance,'High intensity mask':dense,
            'Log':log_correct(equalized),'Gamma':gamma_correct(equalized,gamma)}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Contrast enhancement and pseudocolor.

Task: Enhance an X-ray image so the bones are clearer and apply pseudocolor.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
CLAHE_CLIP = 2.0           # contrast limit; higher = stronger, noisier
CLAHE_TILES = 8            # tiles per side
COLORMAP = cv.COLORMAP_JET # any cv.COLORMAP_* constant


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Local contrast enhancement (CLAHE)
    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
    enhanced = clahe.apply(gray)
    stages['CLAHE'] = enhanced

    # Step 4: Pseudocolor for visual inspection
    colored = cv.applyColorMap(enhanced, COLORMAP)
    stages['Pseudocolor'] = colored

    # Step 5: Measure the contrast gain
    report['contrast_before'] = round(float(gray.std()), 2)
    report['contrast_after'] = round(float(enhanced.std()), 2)
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
