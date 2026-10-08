# Plan: Circle detection and counting with Hough circles

> Count the coins in an image using the Hough circle transform.

## What the task needs

- **Goal:** detect each circle, draw it and count them
- **Input:** one image
- **Method:** Circle detection and counting with Hough circles (chosen by cue words: 'hough circle', 'coins')
- **Also considered:** Geometric transformation (1)
- **Limits to state in your answer:** Parameter meanings differ for other circle methods. Perspective turns circles into ellipses.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Remove speckle noise with a median blur
4. Find circles with the Hough gradient method
5. Draw circles and the count
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

## Step 3: Remove speckle noise with a median blur

A median blur removes salt-and-pepper noise and keeps edges sharp; HoughCircles is very sensitive to noise, so the manual blurs before it.

Parameters:
- `MEDIAN_KSIZE = 5`: odd median kernel size

```python
smooth = cv.medianBlur(gray, MEDIAN_KSIZE)
stages['Median blurred'] = smooth
```

**Watch out:** The kernel size must be odd and greater than 1 (3, 5, 7...).

*Source: Lab 06 Tasks p.3*

## Step 4: Find circles with the Hough gradient method

HOUGH_GRADIENT runs Canny internally (param1 = its high threshold) and votes for centers along gradient directions; param2 is the center-vote threshold (lower = more circles, more false ones). minDist stops one coin being found twice; set it near the smallest coin diameter. The radius range rejects other round things.

Parameters:
- `MIN_DIST = 40`: minimum distance between centers, pixels
- `CANNY_HIGH = 120`: param1: internal Canny high threshold
- `ACC_THRESHOLD = 30`: param2: center votes needed
- `MIN_RADIUS = 10`: pixels
- `MAX_RADIUS = 80`: pixels

```python
found = cv.HoughCircles(smooth, cv.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
                        param1=CANNY_HIGH, param2=ACC_THRESHOLD, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
circles = np.empty((0, 3), int) if found is None else np.round(found[0]).astype(int)
report['count'] = len(circles)
```

**Watch out:** HoughCircles returns None when nothing is found and a (1, N, 3) float array otherwise. Duplicate circles mean MIN_DIST is too small; missed coins mean ACC_THRESHOLD is too high.

**From the course:** Hough circle detection represents a circle as (x - a)^2 + (y - b)^2 = r^2 with center (a, b) and radius r. *(Lab Manual 06 p.18)*

*Source: Lab Manual 06 p.18; Lab 06 Tasks p.3*

## Step 5: Draw circles and the count

Outline + center for each detection, and the total written on the image.

```python
vis = img.copy()
for x, y, r in circles:
    cv.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
cv.putText(vis, f'Count: {len(circles)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19*

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

## Closest worked lab task: Coin counting with Hough circles (L06-T07)

The knowledge base has a tested solution for a similar lab task (match 0.22). Its steps:

1. Grayscale and median blur.
2. Run HoughCircles with plausible radius bounds and center separation.
3. Safely handle no detections.
4. Draw centers/circumferences and count candidates.

Limits: Perspective, touching coins and reflections change performance; tune on labeled actual images.

Code (`solutions/lab06.py`, `lab06.task07_coins`; helpers come from `solutions/cv_core.py`):

```python
def task07_coins(image,min_radius=10,max_radius=80,min_distance=30,param2=25):
    g=cv.medianBlur(gray(image),5)
    circles=cv.HoughCircles(g,cv.HOUGH_GRADIENT,dp=1.2,minDist=min_distance,
                            param1=120,param2=param2,minRadius=min_radius,maxRadius=max_radius)
    found=np.empty((0,3),dtype=int) if circles is None else np.rint(circles[0]).astype(int)
    output=image.copy()
    for x,y,r in found:
        cv.circle(output,(x,y),r,(0,255,0),2); cv.circle(output,(x,y),2,(0,0,255),-1)
    return {'blurred':g,'circles':found,'count':len(found),'overlay':output}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Circle detection and counting with Hough circles.

Task: Count the coins in an image using the Hough circle transform.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
MEDIAN_KSIZE = 5           # odd median kernel size
MIN_DIST = 40              # minimum distance between centers, pixels
CANNY_HIGH = 120           # param1: internal Canny high threshold
ACC_THRESHOLD = 30         # param2: center votes needed
MIN_RADIUS = 10            # pixels
MAX_RADIUS = 80            # pixels


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Remove speckle noise with a median blur
    smooth = cv.medianBlur(gray, MEDIAN_KSIZE)
    stages['Median blurred'] = smooth

    # Step 4: Find circles with the Hough gradient method
    found = cv.HoughCircles(smooth, cv.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
                            param1=CANNY_HIGH, param2=ACC_THRESHOLD, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    circles = np.empty((0, 3), int) if found is None else np.round(found[0]).astype(int)
    report['count'] = len(circles)

    # Step 5: Draw circles and the count
    vis = img.copy()
    for x, y, r in circles:
        cv.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
    cv.putText(vis, f'Count: {len(circles)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
