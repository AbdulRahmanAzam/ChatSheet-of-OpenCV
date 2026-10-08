# Plan: Restricted-zone change monitoring

> Monitor a restricted zone with a camera and raise an alert when someone enters it.

## What the task needs

- **Goal:** detect anything that enters the zone and raise an alert
- **Input:** a video (or webcam)
- **Method:** Restricted-zone change monitoring (chosen by cue words: 'restricted', 'enters', 'zone')
- **Limits to state in your answer:** Needs a static camera and warm-up frames. Lighting changes and moving backgrounds (trees, screens) cause false foreground.

## Steps at a glance

1. Open the video and process it frame by frame
2. Compare each frame with the background
3. Clean the mask with opening and closing
4. Raise an alert when change stays inside the restricted zone
5. Draw zone, moving objects and alert
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
    stages, report = run(frame, state)
```

**Watch out:** Codec support depends on the installed build. This writer creates silent processed video; it does not preserve audio.

*Source: Lab 02 Tasks p.6; Lab 06 Tasks p.2*

## Step 2: Compare each frame with the background

With a static camera, pixels that differ from an empty reference frame by more than DIFF_THRESHOLD are change. Blurring first stops sensor noise from counting; dilation joins broken blobs.

Parameters:
- `DIFF_THRESHOLD = 30`: gray-level change that counts as motion

```python
g = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5, 5), 0)
if 'background' not in state:
    state['background'] = g          # first frame (scene empty) becomes the reference
diff = cv.absdiff(g, state['background'])
_, mask = cv.threshold(diff, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
stages['Difference mask'] = mask
```

**Watch out:** Lighting changes and camera shake also count as change; keep the camera fixed and update the background if light changes slowly.

**From the course:** The Canny example blurs with cv2.GaussianBlur(gray, (5, 5), 1.4) then calls cv2.Canny(blurred, 50, 150): 50 lower threshold, 150 upper threshold. *(Lab Manual 06 p.13)*

**From the course:** cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) converts a color image to grayscale. *(Lab 01 Manual p.15)*

*Source: KB supplement (bgsub)*

## Step 3: Clean the mask with opening and closing

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

## Step 4: Raise an alert when change stays inside the restricted zone

Only change inside the zone polygon matters. Requiring it for PERSIST_FRAMES frames in a row stops one noisy frame from raising an alarm.

Parameters:
- `ZONE = [[100, 100], [300, 100], [300, 300], [100, 300]]`: restricted zone polygon (x, y)
- `MIN_AREA = 200`: smallest changed area, pixels
- `PERSIST_FRAMES = 3`: frames in a row before alert

```python
zone = np.zeros_like(mask)
cv.fillPoly(zone, [np.int32(ZONE)], 255)
inside = cv.bitwise_and(mask, zone)
contours, _ = cv.findContours(inside, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
state['streak'] = state.get('streak', 0) + 1 if contours else 0
report['alert'] = state['streak'] >= PERSIST_FRAMES
```

**Watch out:** The zone polygon is in pixel coordinates of this camera view; redraw it if the camera moves.

*Source: KB supplement (bgsub); Lab 01 Tasks p.3; Lab 05 Manual p.9*

## Step 5: Draw zone, moving objects and alert

Blue = zone, red boxes = changed regions inside it, text when the alert is active.

```python
vis = img.copy()
cv.polylines(vis, [np.int32(ZONE)], True, (255, 0, 0), 2)
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
if report['alert']:
    cv.putText(vis, 'ALERT: zone entered', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19*

## Step 6: Write the annotated frames to an output video

VideoWriter needs the codec, fps and the exact frame size; release both capture and writer at the end.

```python
writer.write(stages.get('Result', frame))   # inside the loop
cap.release(); writer.release()
```

*Source: Lab 02 Tasks p.6; Lab 06 Tasks p.2*

## Closest worked lab task: Restricted-zone visual-change monitor (L06-T08)

The knowledge base has a tested solution for a similar lab task (match 0.3). Its steps:

1. Provide empty background and zone polygon.
2. Compute absolute frame/background difference.
3. Threshold and open the mask; restrict to zone.
4. Filter small components.
5. Require consecutive frames and emit one local event per active episode.

Limits: Explicit policy is any new foreground in the zone; visual change does not prove a person/object is unauthorized. Requires static camera. No external alarm is sent.

Code (`solutions/lab06.py`, `lab06.ZoneMonitor`; helpers come from `solutions/cv_core.py`):

```python
class ZoneMonitor:
    """Task 8. Policy: new foreground in calibrated zone for N consecutive frames.
    Static camera and empty reference frame required. Produces local events only.
    """
    def __init__(self,background,polygon,persistence=3,min_area=100,threshold=30):
        self.background=cv.GaussianBlur(gray(background),(5,5),0)
        self.zone=np.zeros_like(self.background); cv.fillPoly(self.zone,[np.int32(polygon)],255)
        if persistence<1 or min_area<=0 or threshold<0: raise ValueError('Invalid monitor settings')
        self.persistence=persistence; self.min_area=min_area; self.threshold=threshold
        self.count=0; self.active=False
    def update(self,frame):
        g=cv.GaussianBlur(gray(frame),(5,5),0)
        if g.shape!=self.background.shape: raise ValueError('Frame shape changed')
        diff=cv.absdiff(g,self.background)
        mask=cv.threshold(diff,self.threshold,255,cv.THRESH_BINARY)[1]
        mask=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask=cv.bitwise_and(mask,self.zone)
        contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        boxes=[cv.boundingRect(c) for c in contours if cv.contourArea(c)>=self.min_area]
        self.count=self.count+1 if boxes else 0
        active=self.count>=self.persistence; event=active and not self.active; self.active=active
        out=frame.copy()
        for x,y,w,h in boxes: cv.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
        return {'mask':mask,'boxes':boxes,'consecutive_frames':self.count,'active':active,
                'new_event':event,'policy':'new foreground in restricted zone','overlay':out}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Restricted-zone change monitoring.

Task: Monitor a restricted zone with a camera and raise an alert when someone enters it.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
DIFF_THRESHOLD = 30        # gray-level change that counts as motion
MORPH_KSIZE = 5            # structuring element size
ZONE = [[100, 100], [300, 100], [300, 300], [100, 300]] # restricted zone polygon (x, y)
MIN_AREA = 200             # smallest changed area, pixels
PERSIST_FRAMES = 3         # frames in a row before alert


def run(img, state):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Compare each frame with the background
    g = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5, 5), 0)
    if 'background' not in state:
        state['background'] = g          # first frame (scene empty) becomes the reference
    diff = cv.absdiff(g, state['background'])
    _, mask = cv.threshold(diff, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
    mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    stages['Difference mask'] = mask

    # Step 3: Clean the mask with opening and closing
    kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # remove small specks
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # fill small holes
    stages['Cleaned mask'] = mask

    # Step 4: Raise an alert when change stays inside the restricted zone
    zone = np.zeros_like(mask)
    cv.fillPoly(zone, [np.int32(ZONE)], 255)
    inside = cv.bitwise_and(mask, zone)
    contours, _ = cv.findContours(inside, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
    state['streak'] = state.get('streak', 0) + 1 if contours else 0
    report['alert'] = state['streak'] >= PERSIST_FRAMES

    # Step 5: Draw zone, moving objects and alert
    vis = img.copy()
    cv.polylines(vis, [np.int32(ZONE)], True, (255, 0, 0), 2)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if report['alert']:
        cv.putText(vis, 'ALERT: zone entered', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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
    source = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
    cap = cv.VideoCapture(int(source) if source.isdigit() else source)   # 0 = webcam
    if not cap.isOpened():
        raise FileNotFoundError(source)
    fps = cap.get(cv.CAP_PROP_FPS) or 25
    state = {}
    writer, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stages, report = run(frame, state)
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
