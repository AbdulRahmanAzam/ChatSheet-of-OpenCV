# Plan: Basic image operations

> Compute the mean, standard deviation, min and max of each RGB channel.

## What the task needs

- **Goal:** carry out the requested basic operations
- **Input:** one image
- **Method:** Basic image operations (chosen by cue words: 'channel_stats')
- **Limits to state in your answer:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1.

## Steps at a glance

1. Load the image and check it
2. Per-channel statistics
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

## Step 2: Per-channel statistics

cv.split returns channels in B, G, R order.

```python
for name, channel in zip('BGR', cv.split(img)):
    report[f'{name}_mean'] = round(float(channel.mean()), 2)
    report[f'{name}_std'] = round(float(channel.std()), 2)
    report[f'{name}_min_max'] = (int(channel.min()), int(channel.max()))
```

**Watch out:** cv.split is slower than NumPy indexing for a single channel. LUT tables must be uint8 of length 256. np.uint8([-1]) or a cast of out-of-range floats does not provide meaningful saturation. Squaring uint8 before conversion has already lost information.

*Source: KB supplement (border); Lab 04 Manual p.14; Lab Manual 06 p.10*

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

## Closest worked lab task: RGB statistics and optional Pandas describe (L01-T09)

The knowledge base has a tested solution for a similar lab task (match 0.36). Its steps:

1. Convert BGR to RGB and flatten to rows of pixels.
2. Calculate count, mean, sample standard deviation, min, quartiles and max.
3. The optional task09_optional_pandas follows the original DataFrame.describe request.

Limits: Core uses NumPy to respect requested dependency scope; optional Pandas path is preserved but not exercised.

Code (`solutions/lab01.py`, `lab01.task09_rgb_statistics`; helpers come from `solutions/cv_core.py`):

```python
def task09_rgb_statistics(image):
    pixels=cv.cvtColor(image,cv.COLOR_BGR2RGB).reshape(-1,3).astype(float)
    result={}
    for i,name in enumerate(('R','G','B')):
        v=pixels[:,i]; q=np.quantile(v,[0,.25,.5,.75,1])
        result[name]=dict(zip(['count','mean','std','min','25%','50%','75%','max'],
                              [len(v),float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else None,*map(float,q)]))
    return result
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Basic image operations.

Task: Compute the mean, standard deviation, min and max of each RGB channel.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----



def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Per-channel statistics
    for name, channel in zip('BGR', cv.split(img)):
        report[f'{name}_mean'] = round(float(channel.mean()), 2)
        report[f'{name}_std'] = round(float(channel.std()), 2)
        report[f'{name}_min_max'] = (int(channel.min()), int(channel.max()))
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
