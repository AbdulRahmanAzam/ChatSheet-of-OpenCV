# Plan: Basic image operations

> Draw five concentric circles on the image and a bounding box around them.

## What the task needs

- **Goal:** carry out the requested basic operations
- **Input:** one image
- **Method:** Basic image operations (chosen by cue words: 'draw_shapes')
- **Also considered:** Circle detection and counting with Hough circles (1.0)
- **Limits to state in your answer:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1.

## Steps at a glance

1. Load the image and check it
2. Draw shapes and text
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

## Step 2: Draw shapes and text

Drawing functions take points as (x, y), colors as BGR, and thickness -1 to fill.

Parameters:
- `TEXT = 'OpenCV'`: label text

```python
vis = img.copy()
h, w = vis.shape[:2]
cv.rectangle(vis, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 3)
cv.circle(vis, (w // 2, h // 2), min(h, w) // 6, (0, 0, 255), 3)
cv.putText(vis, TEXT, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19*

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

## Closest worked lab task: Five concentric circles and bounding box (L01-T04)

The knowledge base has a tested solution for a similar lab task (match 0.53). Its steps:

1. Create a black 800x800 color canvas.
2. Draw five filled alternating circles from largest to smallest.
3. Draw the outer bounding box and report center/bounds.

Limits: Drawing center (400,400); geometric pixel-center midpoint is (399.5,399.5). Chosen radii 300,240,180,120,60.

Code (`solutions/lab01.py`, `lab01.task04_circles`; helpers come from `solutions/cv_core.py`):

```python
def task04_circles():
    im=np.zeros((800,800,3),np.uint8); center=(400,400)
    for i,radius in enumerate([300,240,180,120,60]):
        cv.circle(im,center,radius,(255,255,255) if i%2==0 else (0,0,0),-1)
    cv.rectangle(im,(100,100),(700,700),(0,255,0),2)
    return {'image':im,'drawing_center':center,'geometric_pixel_center':(399.5,399.5),
            'bounding_box_xyxy':(100,100,700,700)}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Basic image operations.

Task: Draw five concentric circles on the image and a bounding box around them.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
TEXT = 'OpenCV'            # label text


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Draw shapes and text
    vis = img.copy()
    h, w = vis.shape[:2]
    cv.rectangle(vis, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 3)
    cv.circle(vis, (w // 2, h // 2), min(h, w) // 6, (0, 0, 255), 3)
    cv.putText(vis, TEXT, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
