# Plan: Image fusion of two registered images

> Fuse a registered CT slice and MRI slice into one image.

## What the task needs

- **Goal:** blend two aligned images so both kinds of detail are visible
- **Input:** a reference image and a scene image
- **Method:** Image fusion of two registered images (chosen by cue words: 'fuse', 'registered')
- **Also considered:** Contrast enhancement and pseudocolor (1)
- **Limits to state in your answer:** Addition is not a 50/50 blend. Inputs need matching size/type.

## Steps at a glance

1. Load the reference image and the scene image
2. Fuse the two registered images
3. Show every stage side by side and save the result

## Step 1: Load the reference image and the scene image

imread returns None (no exception) for a wrong path, so check it before using the image.

```python
ref = cv.imread('reference.jpg')
img = cv.imread('scene.jpg')
if ref is None or img is None:
    raise FileNotFoundError('check the image paths')
```

**Watch out:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths.

*Source: Lab 01 Manual p.14*

## Step 2: Fuse the two registered images

Weighted blending alpha*A + (1-alpha)*B keeps detail from both images. The images must already be registered (aligned pixel for pixel), otherwise the fusion shows double edges.

Parameters:
- `WEIGHT = 0.5`: weight of the first image (0-1)

```python
if ref.shape != img.shape:
    img = cv.resize(img, (ref.shape[1], ref.shape[0]))   # registered slices must share one size
vis = cv.addWeighted(ref, WEIGHT, img, 1 - WEIGHT, 0)
stages['Fused'] = vis
stages['Result'] = vis
```

**Watch out:** Addition is not a 50/50 blend. Inputs need matching size/type. Different modalities may not share feature appearance. A plausible-looking overlay is not proof of accurate registration.

**From the course:** cv2.resize(image, None, fx=0.5, fy=0.5) scales an image to 50% while keeping the aspect ratio. *(Lab 01 Manual p.16)*

**From the course:** Images must have the same dimensions before they can be added or blended; cv2.resize makes them match. *(Lab 01 Manual p.23)*

*Source: Lab 01 Manual p.23; Lab 02 Tasks p.5; Lab 03 Tasks p.2*

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

## Closest worked lab task: CT/MRI registered-slice fusion (L02-T02)

The knowledge base has a tested solution for a similar lab task (match 0.34). Its steps:

1. Require registered corresponding slices of the same shape.
2. Equalize each grayscale image.
3. Apply distinct colormaps.
4. Blend CT 0.7 and MRI 0.3.
5. Show log and gamma versions.

Limits: registered=True asserts external registration has been performed; this function cannot verify anatomy automatically.

Code (`solutions/lab02.py`, `lab02.task02_fusion`; helpers come from `solutions/cv_core.py`):

```python
def task02_fusion(ct,mri, *, registered, alpha=0.7, gamma=0.6):
    if not registered: raise ValueError('Supply registered corresponding slices; resizing is not registration')
    if ct.shape[:2]!=mri.shape[:2]: raise ValueError('Registered arrays must share shape')
    if not 0<=alpha<=1: raise ValueError('alpha must be in [0,1]')
    a=cv.equalizeHist(gray(ct)); b=cv.equalizeHist(gray(mri))
    ca=cv.applyColorMap(a,cv.COLORMAP_JET); cb=cv.applyColorMap(b,cv.COLORMAP_HOT)
    merged=cv.addWeighted(ca,alpha,cb,1-alpha,0)
    return {'CT equalized':a,'MRI equalized':b,'CT JET':ca,'MRI HOT':cb,
            'CT weighted fusion':merged,'Log fusion':log_correct(merged),'Gamma fusion':gamma_correct(merged,gamma)}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Image fusion of two registered images.

Task: Fuse a registered CT slice and MRI slice into one image.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
WEIGHT = 0.5               # weight of the first image (0-1)


def run(ref, img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Fuse the two registered images
    if ref.shape != img.shape:
        img = cv.resize(img, (ref.shape[1], ref.shape[0]))   # registered slices must share one size
    vis = cv.addWeighted(ref, WEIGHT, img, 1 - WEIGHT, 0)
    stages['Fused'] = vis
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
    ref = cv.imread(sys.argv[1] if len(sys.argv) > 1 else 'reference.jpg')
    img = cv.imread(sys.argv[2] if len(sys.argv) > 2 else 'scene.jpg')
    if ref is None or img is None:
        raise FileNotFoundError('check the reference and scene paths')
    stages, report = run(ref, img)
    for name, value in report.items():
        print(f'{name}: {value}')
    show(img, stages)
    cv.imwrite('result.png', stages['Result'])

if __name__ == '__main__':
    main()
```
