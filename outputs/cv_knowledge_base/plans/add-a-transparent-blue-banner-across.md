# Plan: Basic image operations

> Add a transparent blue banner across the bottom of the image.

## What the task needs

- **Goal:** carry out the requested basic operations
- **Input:** one image
- **Method:** Basic image operations (chosen by cue words: 'blend')
- **Limits to state in your answer:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1.

## Steps at a glance

1. Load the image and check it
2. Blend a transparent banner
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

## Step 2: Blend a transparent banner

addWeighted computes alpha*A + (1-alpha)*B; drawing on a copy and blending gives transparency.

Parameters:
- `ALPHA = 0.4`: banner opacity

```python
vis = img.copy()
h, w = vis.shape[:2]
banner = vis.copy()
cv.rectangle(banner, (0, int(h * 0.85)), (w, h), (255, 0, 0), -1)
vis = cv.addWeighted(banner, ALPHA, vis, 1 - ALPHA, 0)
stages['Result'] = vis
```

**Watch out:** Addition is not a 50/50 blend. Inputs need matching size/type.

*Source: Lab 01 Manual p.23; Lab 02 Tasks p.5*

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

## Closest worked lab task: Transparent blue bottom banner (L01-T06)

The knowledge base has a tested solution for a similar lab task (match 0.48). Its steps:

1. Copy the image and fill the bottom 20 percent with blue.
2. Blend copy/original with opacity and its complement.
3. Add readable text inside the banner.

Limits: Opacity 0.4 and text are configurable. BGR blue=(255,0,0).

Code (`solutions/lab01.py`, `lab01.task06_overlay`; helpers come from `solutions/cv_core.py`):

```python
def task06_overlay(image, opacity=0.4, text='Computer Vision Lab'):
    if not 0<=opacity<=1: raise ValueError('opacity must be in [0,1]')
    overlay=image.copy(); h,w=image.shape[:2]; y=int(h*0.8)
    overlay[y:]=(255,0,0)
    out=cv.addWeighted(overlay,opacity,image,1-opacity,0)
    cv.putText(out,text,(10,min(h-5,y+max(15,(h-y)//2))),cv.FONT_HERSHEY_SIMPLEX,
               max(0.3,w/1000),(255,255,255),1,cv.LINE_AA)
    return out
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Basic image operations.

Task: Add a transparent blue banner across the bottom of the image.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
ALPHA = 0.4                # banner opacity


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Blend a transparent banner
    vis = img.copy()
    h, w = vis.shape[:2]
    banner = vis.copy()
    cv.rectangle(banner, (0, int(h * 0.85)), (w, h), (255, 0, 0), -1)
    vis = cv.addWeighted(banner, ALPHA, vis, 1 - ALPHA, 0)
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
