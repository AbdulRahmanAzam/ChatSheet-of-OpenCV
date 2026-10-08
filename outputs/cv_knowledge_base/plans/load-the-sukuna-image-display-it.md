# Plan: Basic image operations

> Load the Sukuna image, display it with a title and hide the axes.

## What the task needs

- **Goal:** carry out the requested basic operations
- **Input:** one image
- **Method:** Basic image operations (chosen by cue words: 'load')
- **Limits to state in your answer:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
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

## Step 2: Convert to grayscale

Edge, line, circle and threshold operations work on one intensity channel. OpenCV loads color as BGR, so the code must be COLOR_BGR2GRAY (not RGB2GRAY).

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
stages['Grayscale'] = gray
```

**Watch out:** cv.imread gives BGR order; using COLOR_RGB2GRAY swaps the red and blue weights.

**From the course:** cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) converts a color image to grayscale. *(Lab 01 Manual p.15)*

*Source: Lab 01 Manual p.15; Lab 05 Manual p.9*

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

## Closest worked lab task: Load Sukuna image and format display (L01-T03)

The knowledge base has a tested solution for a similar lab task (match 0.41). Its steps:

1. Decode Sukuna.jpeg; fail clearly on missing input.
2. Convert BGR to RGB.
3. Plot at 8x6 inches with axes hidden and dark-red 16-point title with pad 15.

Limits: User must supply Sukuna.jpeg; synthetic demo substitutes a labeled generated image only for testing.

Code (`solutions/lab01.py`, `lab01.task03_display`; helpers come from `solutions/cv_core.py`):

```python
def task03_display(path, output='task03.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    im=read_image(path)
    fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
    ax.set_title('King of Curses — Ryomen Sukuna',fontsize=16,color='darkred',pad=15)
    ax.axis('off'); fig.savefig(output,bbox_inches='tight'); plt.close(fig)
    return im
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Basic image operations.

Task: Load the Sukuna image, display it with a title and hide the axes.

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
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray
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
