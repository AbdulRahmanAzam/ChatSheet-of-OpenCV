# Plan: Color segmentation in HSV

> Isolate the yellow car using HSV.

## What the task needs

- **Goal:** keep only pixels of the target color and outline the objects
- **Input:** one image
- **Method:** Color segmentation in HSV (chosen by cue words: 'hsv', 'yellow', 'isolat')
- **Limits to state in your answer:** 8-bit LAB is scaled (L 0..255, a/b offset by 128), unlike textbook ranges. Skin thresholds are heuristic and vary across people and lighting. Mask dimensions must match the image. AND/OR on arbitrary intensities are bit operations, not probabilistic blending.

## Steps at a glance

1. Load the image and check it
2. Convert to HSV color space
3. Keep pixels inside the color range
4. Clean the mask with opening and closing
5. Find object outlines
6. Draw a box and number around each object
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

## Step 2: Convert to HSV color space

Hue separates "which color" from brightness, so one hue range keeps working in shade and light. OpenCV hue runs 0-179 (not 0-359).

```python
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
```

**Watch out:** 8-bit LAB is scaled (L 0..255, a/b offset by 128), unlike textbook ranges. Skin thresholds are heuristic and vary across people and lighting.

**From the course:** Color thresholding converts to HSV with cv2.cvtColor(image, cv2.COLOR_BGR2HSV) and builds a mask with cv2.inRange(hsv_image, lower_bound, upper_bound). *(Lab 05 Manual p.9)*

*Source: KB supplement (colorspaces)*

## Step 3: Keep pixels inside the color range

inRange keeps pixels whose H, S and V all lie between the bounds. Saturation and value lower bounds stop gray and very dark pixels from matching any hue.

Parameters:
- `HSV_LOW = [20, 100, 100]`: lower H,S,V
- `HSV_HIGH = [35, 255, 255]`: upper H,S,V

```python
mask = cv.inRange(hsv, np.array(HSV_LOW), np.array(HSV_HIGH))
stages['Color mask'] = mask
```

**Watch out:** Red hue wraps around 0/180, so full red needs two ranges joined with cv.bitwise_or.

*Source: KB supplement (colorspaces); Lab 01 Tasks p.3; Lab 05 Manual p.9*

## Step 4: Clean the mask with opening and closing

Opening (erode then dilate) removes specks smaller than the kernel; closing (dilate then erode) fills small holes and gaps. Kernel larger than the noise, smaller than the objects.

Parameters:
- `MORPH_KSIZE = 5`: structuring element size

```python
kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # remove small specks
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # fill small holes
stages['Cleaned mask'] = mask
```

**Watch out:** A kernel bigger than the smallest object deletes it.

*Source: Lab 05 Manual p.13; Lab 05 Tasks p.7*

## Step 5: Find object outlines

findContours traces the boundary of each white blob; RETR_EXTERNAL keeps outer boundaries only. Small areas are noise, so they are dropped with MIN_AREA.

Parameters:
- `MIN_AREA = 100`: smallest object area in pixels

```python
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
report['count'] = len(contours)
report['areas'] = [int(cv.contourArea(c)) for c in contours]
```

**Watch out:** findContours expects white objects on black; invert the mask first if your objects are dark.

*Source: KB supplement (contours); Lab 06 Tasks p.1*

## Step 6: Draw a box and number around each object

Bounding boxes show where each object is and the number shows the count.

```python
vis = img.copy()
for i, c in enumerate(contours, 1):
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(vis, str(i), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center. RETR_EXTERNAL ignores nested holes. A bounding rectangle is not a perspective quadrilateral; contours and bounding-box pixel areas differ.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19; KB supplement (contours); Lab 06 Tasks p.1*

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

## Closest worked lab task: HSV yellow-car isolation (L05-T04)

The knowledge base has a tested solution for a similar lab task (match 0.42). Its steps:

1. Convert BGR to HSV.
2. Create a narrow yellow mask and broader yellow mask.
3. Extract pixels with each mask.
4. Inspect background leakage versus missed dim yellow areas.

Limits: Fixed ranges are starting examples, not calibrated for missing car images. Hue is 0..179 for uint8.

Code (`solutions/lab05.py`, `lab05.task04_yellow`; helpers come from `solutions/cv_core.py`):

```python
def task04_yellow(image):
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    narrow=cv.inRange(hsv,np.array([22,120,100],np.uint8),np.array([32,255,255],np.uint8))
    broad=cv.inRange(hsv,np.array([15,60,50],np.uint8),np.array([40,255,255],np.uint8))
    return {'Original':image,'Narrow mask':narrow,'Narrow extraction':cv.bitwise_and(image,image,mask=narrow),
            'Broad mask':broad,'Broad extraction':cv.bitwise_and(image,image,mask=broad)}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Color segmentation in HSV.

Task: Isolate the yellow car using HSV.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
HSV_LOW = [20, 100, 100]   # lower H,S,V
HSV_HIGH = [35, 255, 255]  # upper H,S,V
MORPH_KSIZE = 5            # structuring element size
MIN_AREA = 100             # smallest object area in pixels


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Step 3: Keep pixels inside the color range
    mask = cv.inRange(hsv, np.array(HSV_LOW), np.array(HSV_HIGH))
    stages['Color mask'] = mask

    # Step 4: Clean the mask with opening and closing
    kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # remove small specks
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # fill small holes
    stages['Cleaned mask'] = mask

    # Step 5: Find object outlines
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
    report['count'] = len(contours)
    report['areas'] = [int(cv.contourArea(c)) for c in contours]

    # Step 6: Draw a box and number around each object
    vis = img.copy()
    for i, c in enumerate(contours, 1):
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(vis, str(i), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
