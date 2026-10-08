# Plan: Contrast enhancement and pseudocolor

> Visualize an echocardiogram video with better contrast.

## What the task needs

- **Goal:** make the important detail visible and measure the improvement
- **Input:** one image processed frame by frame from a video or webcam
- **Method:** Contrast enhancement and pseudocolor (chosen by cue words: 'contrast', 'echo')
- **Limits to state in your answer:** equalizeHist expects uint8 single-channel input. Independently equalizing BGR channels alters colors; work on a luminance channel when color fidelity matters. Some libraries define a reciprocal gamma parameter; state the convention. Convert before arithmetic.

## Steps at a glance

1. Open the video and process it frame by frame
2. Convert to grayscale
3. Local contrast enhancement (CLAHE)
4. Pseudocolor for visual inspection
5. Measure the contrast gain
6. Write the annotated frames to an output video

## Step 1: Open the video and process it frame by frame

A video is a sequence of images. Each frame goes through the same steps; check isOpened() and stop when read() returns False. Pass 0 instead of a file name for a webcam.

```python
cap = cv.VideoCapture(source)            # file name, or 0 for the webcam
if not cap.isOpened():
    raise FileNotFoundError(source)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    stages, report = run(frame)
```

**Watch out:** Codec support depends on the installed build. This writer creates silent processed video; it does not preserve audio.

*Source: Lab 02 Tasks p.6; Lab 06 Tasks p.2*

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

## Step 6: Write the annotated frames to an output video

VideoWriter needs the codec, fps and the exact frame size; release both capture and writer at the end.

```python
writer.write(stages.get('Result', frame))   # inside the loop
cap.release(); writer.release()
```

*Source: Lab 02 Tasks p.6; Lab 06 Tasks p.2*

## Closest worked lab task: Echo video visualization (L02-T03)

The knowledge base has a tested solution for a similar lab task (match 0.18). Its steps:

1. Open local video and validate decoder.
2. Process each frame with grayscale equalization, intensity pseudocolor, gains, log and gamma.
3. Concatenate original and processed views.
4. Save a silent video and release resources.

Limits: Colors visualize intensity; not Doppler blood flow. Dataset links preserved in source pages; no dataset downloaded.

Code (`solutions/lab02.py`, `lab02.task03_echo_video`; helpers come from `solutions/cv_core.py`):

```python
def task03_echo_video(path,output_path):
    return process_video(path,task03_echo_frame,output_path)
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Contrast enhancement and pseudocolor.

Task: Visualize an echocardiogram video with better contrast.

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
    source = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
    cap = cv.VideoCapture(int(source) if source.isdigit() else source)   # 0 = webcam
    if not cap.isOpened():
        raise FileNotFoundError(source)
    fps = cap.get(cv.CAP_PROP_FPS) or 25
    writer, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stages, report = run(frame)
        out = stages.get('Result', frame)
        if writer is None:
            writer = cv.VideoWriter('result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, out.shape[1::-1])
        writer.write(out)
        n += 1
        if n % 25 == 1 or report.get('alert'):
            print(f'frame {n}:', {k: v for k, v in report.items() if not isinstance(v, (list, np.ndarray))})
    cap.release()
    if writer is not None:
        writer.release()
    print(f'{n} frames written to result.mp4')

if __name__ == '__main__':
    main()
```
