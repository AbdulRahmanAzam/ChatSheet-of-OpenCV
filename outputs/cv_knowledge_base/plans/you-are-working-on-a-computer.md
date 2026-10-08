# Plan: Screen detection and status with Hough lines

> You are working on a computer vision project for monitoring computer lab usage and ensuring the availability of computer screens. Your goal is to implement screen detection using the Hough Line Transformation. The lab contains rows of computers, and you need to identify the boundaries of computer screens to monitor their status (on or off) and detect any anomalies, such as missing screens.

## What the task needs

- **Goal:** find every screen, say whether it is ON or OFF, and report missing screens
- **Input:** one image
- **Method:** Screen detection and status with Hough lines (chosen by cue words: 'screens', 'on or off', 'computer lab', 'missing')
- **Also considered:** Straight-line detection with Hough (2), Object recognition with SIFT, ratio test and RANSAC homography (2), Wavelet denoising and anomaly detection on a signal (2)
- **Limits to state in your answer:** Assumes a roughly front-facing camera so screen borders are near horizontal/vertical. Brightness shows whether a screen is lit, not whether the computer is powered (a dark desktop looks OFF, glare looks ON). A missing screen is inferred from a gap in the row/column layout, so a person blocking a screen also looks missing.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Smooth noise with a Gaussian blur
4. Detect edges with Canny
5. Find line segments with probabilistic Hough
6. Keep horizontal and vertical lines only
7. Draw the Hough lines on a blank canvas
8. Turn line outlines into screen rectangles
9. Decide ON / OFF from brightness inside each screen
10. Find missing screens from the row/column layout
11. Draw screens, states and missing slots
12. Show every stage side by side and save the result

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

## Step 3: Smooth noise with a Gaussian blur

Noise creates false edges and false Hough votes. A Gaussian blur removes fine noise while keeping strong boundaries. Kernel size must be odd; bigger = smoother but weaker thin edges.

Parameters:
- `BLUR_KSIZE = 7`: odd Gaussian kernel size; raise to 7-9 for noisy images

```python
smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
stages['Blurred'] = smooth
```

**Watch out:** Too much blur merges nearby edges (two screens, two lane dashes); too little lets noise vote.

**From the course:** cv2.GaussianBlur(image, (31, 31), 0) blurs with a 31x31 kernel; kernel sizes must be odd numbers. (Note: The code comment mentions 5x5 but the code uses 31x31 (correction C03).) *(Lab 01 Manual p.17)*

**From the course:** Gaussian blur uses a Gaussian-shaped kernel and gives smoother results than box blur; often used for noise reduction. *(Lab 04 Manual p.17)*

*Source: Lab 01 Manual p.17; Lab 04 Manual p.17*

## Step 4: Detect edges with Canny

Canny gives thin, connected edges (gradient, non-maximum suppression, hysteresis). Pixels above CANNY_HIGH are sure edges; pixels between the two thresholds survive only if connected to a sure edge. A 1:2 to 1:3 ratio is the usual choice. Raise both if texture gives too many edges.

Parameters:
- `CANNY_LOW = 50`: hysteresis low threshold
- `CANNY_HIGH = 150`: hysteresis high threshold

```python
edges = cv.Canny(smooth, CANNY_LOW, CANNY_HIGH)
stages['Canny edges'] = edges
```

**Watch out:** Too low thresholds give texture edges that later vote for false lines/circles; too high break real boundaries.

**From the course:** Hysteresis thresholding (Canny) uses two thresholds: a high threshold to detect strong edges and a low threshold to link weak edges; example cv2.Canny(image, 100, 200). *(Lab 05 Manual p.10)*

**From the course:** Canny edge detection steps: Gaussian smoothing reduces noise, gradient calculation finds magnitude and direction, non-maximum suppression, and hysteresis thresholding. *(Lab Manual 06 p.11)*

*Source: Lab 04 Manual p.22; Lab Manual 06 p.11; Lab Manual 06 p.13*

## Step 5: Find line segments with probabilistic Hough

Each edge pixel votes for all lines (rho, theta) through it; cells with at least HOUGH_THRESHOLD votes are lines. HoughLinesP returns finite segments (x1,y1,x2,y2), which is what boundaries need. minLineLength drops short texture lines, maxLineGap joins broken edges. It returns None when nothing is found.

Parameters:
- `HOUGH_THRESHOLD = 40`: minimum votes (accumulator threshold)
- `MIN_LINE_LENGTH = 40`: shortest segment kept, pixels
- `MAX_LINE_GAP = 25`: largest gap joined into one segment, pixels

```python
found = cv.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                       minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
lines = np.empty((0, 4), int) if found is None else found.reshape(-1, 4)
report['line_segments'] = len(lines)
```

**Watch out:** HoughLinesP returns None when nothing passes the threshold, so check before indexing. Noisy edges break one side into several segments; raise MAX_LINE_GAP to join them.

**From the course:** Hough voting: for each edge pixel (typically from Canny) increment every parameter cell of lines passing through it. *(Lab Manual 06 p.17)*

*Source: Lab Manual 06 p.17; Lab 06 Tasks p.1; Lab 06 Tasks p.2*

## Step 6: Keep horizontal and vertical lines only

Screen borders are horizontal and vertical in a front view; dropping slanted lines removes keyboards, cables and chair edges.

Parameters:
- `ANGLE_TOLERANCE = 10`: degrees a line may lean and still count as horizontal/vertical

```python
angle = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180
horizontal = np.minimum(angle, 180 - angle) < ANGLE_TOLERANCE
vertical = np.abs(angle - 90) < ANGLE_TOLERANCE
lines = lines[horizontal | vertical]
report['horizontal_lines'] = int(horizontal.sum())
report['vertical_lines'] = int(vertical.sum())
```

**Watch out:** For a camera looking at the screens from an angle, borders are no longer horizontal/vertical; raise ANGLE_TOLERANCE or rectify the view first.

*Source: Lab Manual 06 p.17; Lab 06 Tasks p.1; Lab 06 Tasks p.2*

## Step 7: Draw the Hough lines on a blank canvas

Rebuilding the image from lines only keeps the straight structure and drops curved clutter; closing bridges small gaps where two segments almost meet, so the sides of one screen join into one shape.

Parameters:
- `CLOSE_KSIZE = 9`: closing kernel that bridges gaps between segments

```python
canvas = np.zeros_like(edges)
for x1, y1, x2, y2 in lines:
    cv.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
canvas = cv.morphologyEx(canvas, cv.MORPH_CLOSE, np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8))
stages['Hough lines'] = canvas
```

**Watch out:** A closing kernel larger than the gap between neighbouring screens merges them into one shape.

**From the course:** The Hough transform maps the image spatial domain into a parameter space where each pixel corresponds to a curve or point. *(Lab Manual 06 p.17)*

**From the course:** The Hough transform detects simple shapes within an image, most commonly lines and circles. *(Lab Manual 06 p.17)*

*Source: Lab Manual 06 p.17; Lab 06 Tasks p.1; Lab 06 Tasks p.2; Lab 05 Manual p.13*

## Step 8: Turn line outlines into screen rectangles

A screen is a rectangle of the right size and shape whose sides are made of Hough lines. For each outline the bounding box is tested: at least 3 of its 4 sides must be covered by line pixels (SIDE_SUPPORT). The area and width/height filter rejects keyboards, desks and posters; boxes inside a bigger box (the inner screen edge inside the bezel) are dropped so each monitor is counted once.

Parameters:
- `MIN_SCREEN_AREA = 2000`: smallest screen area in pixels
- `MIN_ASPECT = 1.0`: smallest width/height
- `MAX_ASPECT = 2.5`: largest width/height
- `SIDE_SUPPORT = 0.6`: share of a side that must lie on Hough lines

```python
contours, _ = cv.findContours(canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
screens = []
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    if w * h < MIN_SCREEN_AREA or not MIN_ASPECT <= w / h <= MAX_ASPECT:
        continue
    sides = side_support(canvas, (x, y, w, h))           # share of each side covered by Hough lines
    if sum(s >= SIDE_SUPPORT for s in sides) >= 3:       # one side may be hidden (person, cable, glare)
        screens.append((x, y, w, h))
screens = sort_rows(drop_nested(screens))
report['screens_found'] = len(screens)

# helper used above (defined once, outside run):
def side_support(canvas, box, band=4):
    """Fraction of each side (top, bottom, left, right) of box that has line pixels within band px."""
    x, y, w, h = box
    near = cv.dilate(canvas, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0
    y2, x2 = min(y + h - 1, canvas.shape[0] - 1), min(x + w - 1, canvas.shape[1] - 1)
    return [near[y, x:x2 + 1].mean(), near[y2, x:x2 + 1].mean(), near[y:y2 + 1, x].mean(), near[y:y2 + 1, x2].mean()]


def drop_nested(boxes):
    """Remove boxes that lie inside another (bigger) box."""
    inside = lambda a, b: a != b and a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]
    return [a for a in boxes if not any(inside(a, b) for b in boxes)]


def sort_rows(boxes):
    """Order boxes row by row (top to bottom), left to right inside a row."""
    if not boxes:
        return []
    row_height = np.median([b[3] for b in boxes])
    return sorted(boxes, key=lambda b: (int(round(b[1] / row_height)), b[0]))
```

**Watch out:** A bounding rectangle is axis-aligned; for tilted screens use cv.minAreaRect instead.

*Source: KB supplement (contours); Lab 06 Tasks p.1*

## Step 9: Decide ON / OFF from brightness inside each screen

A lit screen is much brighter than a dark one. The center of the box (inset 20%) is measured so the bezel does not count. Choose ON_LEVEL between the two groups seen in your image (look at the printed brightness).

Parameters:
- `ON_LEVEL = 100`: mean gray level above which a screen counts as ON

```python
states = []
for x, y, w, h in screens:
    inner = gray[y + h // 5: y + h - h // 5, x + w // 5: x + w - w // 5]   # skip the bezel
    level = float(inner.mean())
    states.append({'box': (x, y, w, h), 'brightness': round(level, 1),
                   'state': 'ON' if level > ON_LEVEL else 'OFF'})
report['on'] = sum(s['state'] == 'ON' for s in states)
report['off'] = sum(s['state'] == 'OFF' for s in states)
```

**Watch out:** Pick ON_LEVEL from your own image: print the brightness values and choose a value between the dark and the lit group.

*Source: Lab 06 Tasks p.1; Lab 04 Manual p.9; Lab 05 Tasks p.3*

## Step 10: Find missing screens from the row/column layout

Computers stand in rows and columns. The centers of the found screens give the row and column positions; any (row, column) cell with no screen is reported missing. If you know the expected count, set EXPECTED_SCREENS so a whole missing row/column is also caught.

Parameters:
- `EXPECTED_SCREENS = None`: expected number of screens, or None to infer from the layout

```python
missing = find_missing(screens, EXPECTED_SCREENS)
report['missing'] = len(missing)
report['missing_positions'] = missing

# helper used above (defined once, outside run):
def cluster_1d(values, gap):
    """Group sorted numbers whose neighbours are closer than gap; return the group means."""
    groups = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] < gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def find_missing(boxes, expected=None):
    """Grid cells (row, col, x, y) that have no box. Assumes screens stand in rows and columns."""
    if not boxes:
        return []
    w = np.median([b[2] for b in boxes]); h = np.median([b[3] for b in boxes])
    centers = [(x + bw / 2, y + bh / 2) for x, y, bw, bh in boxes]
    cols = cluster_1d([c[0] for c in centers], w / 2)
    rows = cluster_1d([c[1] for c in centers], h / 2)
    missing = []
    for r, cy in enumerate(rows):
        for c, cx in enumerate(cols):
            if not any(abs(px - cx) < w / 2 and abs(py - cy) < h / 2 for px, py in centers):
                missing.append((r, c, int(cx), int(cy)))
    if expected is not None and len(boxes) + len(missing) < expected:
        missing.append(('unknown', 'unknown', expected - len(boxes) - len(missing), 'not on the grid'))
    return missing
```

**Watch out:** The layout is learned from the screens that were found, so a whole missing row or column is only caught when EXPECTED_SCREENS is set.

*Source: Lab 06 Tasks p.1*

## Step 11: Draw screens, states and missing slots

The annotated image is the deliverable: green = ON, blue = OFF, red cross = missing screen.

```python
vis = img.copy()
for s in states:
    x, y, w, h = s['box']
    color = (0, 255, 0) if s['state'] == 'ON' else (255, 128, 0)
    cv.rectangle(vis, (x, y), (x + w, y + h), color, 3)
    cv.putText(vis, s['state'], (x + 5, y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
for m in missing:
    if isinstance(m[2], int) and isinstance(m[3], int):
        cv.drawMarker(vis, (m[2], m[3]), (0, 0, 255), cv.MARKER_TILTED_CROSS, 60, 4)
        cv.putText(vis, 'MISSING', (m[2] - 50, m[3] + 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center.

**From the course:** cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. *(Lab 01 Manual p.19)*

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19*

## Step 12: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: Hough-supported screen monitoring (L06-T01)

The knowledge base has a tested solution for a similar lab task (match 0.37). Its steps:

1. Read image, grayscale, smooth and Canny.
2. Detect Hough line segments and draw them on a black canvas.
3. Check support around expected screen rectangles.
4. Measure interior brightness.
5. Display supported boundaries and unconfirmed locations.

Limits: Requires calibrated front-facing ROIs. Outputs visible_dark/visible_bright/unconfirmed; absence may be missing or occluded, brightness is not proven power state.

Code (`solutions/lab06.py`, `lab06.task01_screens`; helpers come from `solutions/cv_core.py`):

```python
def task01_screens(image,expected_rectangles,edge_support=0.35,dark_threshold=35):
    """Front-facing, calibrated screen ROIs [(x,y,w,h),...]. Hough-supported sides.
    Output is visible/dark/bright or unconfirmed; 'missing' cannot be proven from
    absent lines alone (occlusion/glare also remove edges). No inferred power state.
    """
    edges,lines=hough_segments(image); h,w=edges.shape
    line_canvas=np.zeros_like(edges)
    for x1,y1,x2,y2 in lines: cv.line(line_canvas,(x1,y1),(x2,y2),255,2)
    # Tolerance band accounts for double edges and small calibration errors.
    support=cv.dilate(line_canvas,np.ones((7,7),np.uint8))
    annotated=image.copy(); results=[]
    for index,(x,y,rw,rh) in enumerate(expected_rectangles):
        x,y,rw,rh=map(int,(x,y,rw,rh))
        if rw<12 or rh<12 or x<0 or y<0 or x+rw>w or y+rh>h: raise ValueError('Invalid calibrated ROI')
        x2,y2=x+rw-1,y+rh-1
        side_support=[np.mean(support[y,x:x+rw]>0),np.mean(support[y2,x:x+rw]>0),
                      np.mean(support[y:y+rh,x]>0),np.mean(support[y:y+rh,x2]>0)]
        visible=sum(s>=edge_support for s in side_support)>=3
        brightness=float(np.mean(gray(image)[y+5:y2-4,x+5:x2-4]))
        state=('visible_dark' if brightness<dark_threshold else 'visible_bright') if visible else 'unconfirmed_possible_missing_or_occluded'
        results.append({'screen':index,'state':state,'side_support':list(map(float,side_support)),
                        'mean_interior_intensity':brightness,'expected_xywh':[x,y,rw,rh]})
        color=(0,255,0) if visible else (0,0,255)
        cv.rectangle(annotated,(x,y),(x2,y2),color,2)
        cv.putText(annotated,f'{index}: '+('dark' if visible and brightness<dark_threshold else 'bright' if visible else 'check'),
                   (x,y-5 if y>15 else y+15),cv.FONT_HERSHEY_SIMPLEX,.4,color,1)
    return {'gray':gray(image),'edges':edges,'lines_on_black':line_canvas,'overlay':annotated,'screens':results}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Screen detection and status with Hough lines.

Task: You are working on a computer vision project for monitoring computer lab usage and ensuring the
      availability of computer screens. Your goal is to implement screen detection using the Hough
      Line Transformation. The lab contains rows of computers, and you need to identify the
      boundaries of computer screens to monitor their status (on or off) and detect any anomalies,
      such as missing screens.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 7             # odd Gaussian kernel size; raise to 7-9 for noisy images
CANNY_LOW = 50             # hysteresis low threshold
CANNY_HIGH = 150           # hysteresis high threshold
HOUGH_THRESHOLD = 40       # minimum votes (accumulator threshold)
MIN_LINE_LENGTH = 40       # shortest segment kept, pixels
MAX_LINE_GAP = 25          # largest gap joined into one segment, pixels
ANGLE_TOLERANCE = 10       # degrees a line may lean and still count as horizontal/vertical
CLOSE_KSIZE = 9            # closing kernel that bridges gaps between segments
MIN_SCREEN_AREA = 2000     # smallest screen area in pixels
MIN_ASPECT = 1.0           # smallest width/height
MAX_ASPECT = 2.5           # largest width/height
SIDE_SUPPORT = 0.6         # share of a side that must lie on Hough lines
ON_LEVEL = 100             # mean gray level above which a screen counts as ON
EXPECTED_SCREENS = None    # expected number of screens, or None to infer from the layout


def side_support(canvas, box, band=4):
    """Fraction of each side (top, bottom, left, right) of box that has line pixels within band px."""
    x, y, w, h = box
    near = cv.dilate(canvas, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0
    y2, x2 = min(y + h - 1, canvas.shape[0] - 1), min(x + w - 1, canvas.shape[1] - 1)
    return [near[y, x:x2 + 1].mean(), near[y2, x:x2 + 1].mean(), near[y:y2 + 1, x].mean(), near[y:y2 + 1, x2].mean()]


def drop_nested(boxes):
    """Remove boxes that lie inside another (bigger) box."""
    inside = lambda a, b: a != b and a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]
    return [a for a in boxes if not any(inside(a, b) for b in boxes)]


def sort_rows(boxes):
    """Order boxes row by row (top to bottom), left to right inside a row."""
    if not boxes:
        return []
    row_height = np.median([b[3] for b in boxes])
    return sorted(boxes, key=lambda b: (int(round(b[1] / row_height)), b[0]))


def cluster_1d(values, gap):
    """Group sorted numbers whose neighbours are closer than gap; return the group means."""
    groups = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] < gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def find_missing(boxes, expected=None):
    """Grid cells (row, col, x, y) that have no box. Assumes screens stand in rows and columns."""
    if not boxes:
        return []
    w = np.median([b[2] for b in boxes]); h = np.median([b[3] for b in boxes])
    centers = [(x + bw / 2, y + bh / 2) for x, y, bw, bh in boxes]
    cols = cluster_1d([c[0] for c in centers], w / 2)
    rows = cluster_1d([c[1] for c in centers], h / 2)
    missing = []
    for r, cy in enumerate(rows):
        for c, cx in enumerate(cols):
            if not any(abs(px - cx) < w / 2 and abs(py - cy) < h / 2 for px, py in centers):
                missing.append((r, c, int(cx), int(cy)))
    if expected is not None and len(boxes) + len(missing) < expected:
        missing.append(('unknown', 'unknown', expected - len(boxes) - len(missing), 'not on the grid'))
    return missing


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Smooth noise with a Gaussian blur
    smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    stages['Blurred'] = smooth

    # Step 4: Detect edges with Canny
    edges = cv.Canny(smooth, CANNY_LOW, CANNY_HIGH)
    stages['Canny edges'] = edges

    # Step 5: Find line segments with probabilistic Hough
    found = cv.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                           minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    lines = np.empty((0, 4), int) if found is None else found.reshape(-1, 4)
    report['line_segments'] = len(lines)

    # Step 6: Keep horizontal and vertical lines only
    angle = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180
    horizontal = np.minimum(angle, 180 - angle) < ANGLE_TOLERANCE
    vertical = np.abs(angle - 90) < ANGLE_TOLERANCE
    lines = lines[horizontal | vertical]
    report['horizontal_lines'] = int(horizontal.sum())
    report['vertical_lines'] = int(vertical.sum())

    # Step 7: Draw the Hough lines on a blank canvas
    canvas = np.zeros_like(edges)
    for x1, y1, x2, y2 in lines:
        cv.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
    canvas = cv.morphologyEx(canvas, cv.MORPH_CLOSE, np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8))
    stages['Hough lines'] = canvas

    # Step 8: Turn line outlines into screen rectangles
    contours, _ = cv.findContours(canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    screens = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w * h < MIN_SCREEN_AREA or not MIN_ASPECT <= w / h <= MAX_ASPECT:
            continue
        sides = side_support(canvas, (x, y, w, h))           # share of each side covered by Hough lines
        if sum(s >= SIDE_SUPPORT for s in sides) >= 3:       # one side may be hidden (person, cable, glare)
            screens.append((x, y, w, h))
    screens = sort_rows(drop_nested(screens))
    report['screens_found'] = len(screens)

    # Step 9: Decide ON / OFF from brightness inside each screen
    states = []
    for x, y, w, h in screens:
        inner = gray[y + h // 5: y + h - h // 5, x + w // 5: x + w - w // 5]   # skip the bezel
        level = float(inner.mean())
        states.append({'box': (x, y, w, h), 'brightness': round(level, 1),
                       'state': 'ON' if level > ON_LEVEL else 'OFF'})
    report['on'] = sum(s['state'] == 'ON' for s in states)
    report['off'] = sum(s['state'] == 'OFF' for s in states)

    # Step 10: Find missing screens from the row/column layout
    missing = find_missing(screens, EXPECTED_SCREENS)
    report['missing'] = len(missing)
    report['missing_positions'] = missing

    # Step 11: Draw screens, states and missing slots
    vis = img.copy()
    for s in states:
        x, y, w, h = s['box']
        color = (0, 255, 0) if s['state'] == 'ON' else (255, 128, 0)
        cv.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        cv.putText(vis, s['state'], (x + 5, y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for m in missing:
        if isinstance(m[2], int) and isinstance(m[3], int):
            cv.drawMarker(vis, (m[2], m[3]), (0, 0, 255), cv.MARKER_TILTED_CROSS, 60, 4)
            cv.putText(vis, 'MISSING', (m[2] - 50, m[3] + 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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
