# Plan: Lane-line detection with Hough lines

> You are working on an autonomous vehicle project, and one of the critical tasks is to detect lane markings on the road to ensure safe driving, using the Hough Line Transformation.

## What the task needs

- **Goal:** find the left and right lane lines and draw them
- **Input:** one image
- **Method:** Lane-line detection with Hough lines (chosen by cue words: 'lane', 'lane markings', 'autonomous')
- **Also considered:** Straight-line detection with Hough (2), Geometric transformation (1)
- **Limits to state in your answer:** A Hough line is not automatically a screen or lane. Texture can vote strongly, repeated edges cause duplicates, and lines can be absent.

## Steps at a glance

1. Load the image and check it
2. Convert to grayscale
3. Smooth noise with a Gaussian blur
4. Detect edges with Canny
5. Keep only the road region in front of the car
6. Find line segments with probabilistic Hough
7. Split left/right lane segments and fit one line each
8. Draw the lanes on the frame
9. Show every stage side by side and save the result

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
- `BLUR_KSIZE = 5`: odd Gaussian kernel size; raise to 7-9 for noisy images

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

## Step 5: Keep only the road region in front of the car

Lane markings sit in a trapezoid in front of the car; masking everything else removes trees, sky and other cars before voting.

Parameters:
- `ROI_TOP = 0.6`: top of the road trapezoid as a fraction of image height

```python
h, w = edges.shape
polygon = np.int32([[(0, h - 1), (int(w * 0.45), int(h * ROI_TOP)), (int(w * 0.55), int(h * ROI_TOP)), (w - 1, h - 1)]])
roi = np.zeros_like(edges)
cv.fillPoly(roi, polygon, 255)
edges = cv.bitwise_and(edges, roi)
stages['Edges in road ROI'] = edges
```

**Watch out:** If the camera position changes, the trapezoid must be moved too, or lane pixels get cut off.

*Source: Lab 01 Tasks p.3; Lab 05 Manual p.9; Lab Manual 06 p.17; Lab 06 Tasks p.1*

## Step 6: Find line segments with probabilistic Hough

Each edge pixel votes for all lines (rho, theta) through it; cells with at least HOUGH_THRESHOLD votes are lines. HoughLinesP returns finite segments (x1,y1,x2,y2), which is what boundaries need. minLineLength drops short texture lines, maxLineGap joins broken edges. It returns None when nothing is found.

Parameters:
- `HOUGH_THRESHOLD = 20`: minimum votes (accumulator threshold)
- `MIN_LINE_LENGTH = 20`: shortest segment kept, pixels
- `MAX_LINE_GAP = 60`: largest gap joined into one segment, pixels

```python
found = cv.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                       minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
lines = np.empty((0, 4), int) if found is None else found.reshape(-1, 4)
report['line_segments'] = len(lines)
```

**Watch out:** HoughLinesP returns None when nothing passes the threshold, so check before indexing. Noisy edges break one side into several segments; raise MAX_LINE_GAP to join them.

**From the course:** Hough voting: for each edge pixel (typically from Canny) increment every parameter cell of lines passing through it. *(Lab Manual 06 p.17)*

*Source: Lab Manual 06 p.17; Lab 06 Tasks p.1; Lab 06 Tasks p.2*

## Step 7: Split left/right lane segments and fit one line each

In image coordinates (y grows downward) the left lane has negative slope and the right lane positive slope. Fitting x = a*y + b to all points of one side averages the dashed pieces into one stable lane line.

Parameters:
- `MIN_SLOPE = 0.4`: ignore segments flatter than this |slope|

```python
h, w = edges.shape
lanes = {}
for side in ('left', 'right'):
    pts = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE:
            continue                      # nearly horizontal: not a lane marking
        if (side == 'left') == (slope < 0):
            pts += [(x1, y1), (x2, y2)]
    if len(pts) >= 2:
        p = np.array(pts, float)
        a, b = np.polyfit(p[:, 1], p[:, 0], 1)          # x = a*y + b (stable for steep lines)
        ys = np.array([h - 1, int(h * ROI_TOP)])
        lanes[side] = [(int(a * yy + b), int(yy)) for yy in ys]
report['lanes_found'] = sorted(lanes)
```

**Watch out:** Fit x as a function of y; near-vertical lanes make y = m*x + c blow up.

**From the course:** For line detection the manual represents a line as y = mx + b, with slope m and y-intercept b as parameter space axes. (Note: OpenCV uses the polar form rho = x cos(theta) + y sin(theta), which handles vertical lines (correction C26).) *(Lab Manual 06 p.17)*

**From the course:** The Hough transform maps the image spatial domain into a parameter space where each pixel corresponds to a curve or point. *(Lab Manual 06 p.17)*

*Source: Lab Manual 06 p.17; Lab 06 Tasks p.1; Lab 06 Tasks p.2*

## Step 8: Draw the lanes on the frame

Lines are drawn on a separate layer and blended so the road stays visible under them.

```python
vis = img.copy()
overlay = np.zeros_like(img)
for (p1, p2) in lanes.values():
    cv.line(overlay, p1, p2, (0, 255, 0), 10)
vis = cv.addWeighted(vis, 1.0, overlay, 0.8, 0)
stages['Result'] = vis
```

**Watch out:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center. Addition is not a 50/50 blend. Inputs need matching size/type.

*Source: Lab 01 Tasks p.2; Lab 01 Manual p.19; Lab 01 Manual p.23; Lab 02 Tasks p.5*

## Step 9: Show every stage side by side and save the result

Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". The report (counts, states, thresholds) is printed.

```python
for name, value in report.items():
    print(f'{name}: {value}')
show(img, stages)
cv.imwrite('result.png', stages['Result'])
```

**Watch out:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: Lab 01 Manual p.14; Lab Manual 06 p.10; KB supplement (matplotlib)*

## Closest worked lab task: Hough lane-line detection (L06-T06)

The knowledge base has a tested solution for a similar lab task (match 0.26). Its steps:

1. Smooth and detect edges.
2. Restrict to a road-shaped trapezoid.
3. Detect segments and group by slope/side.
4. Fit x as a function of y and draw left/right boundaries.

Limits: Fixed-camera straight-lane teaching example, not a vehicle-control system. ROI/slope assumptions fail on curves/hills.

Code (`solutions/lab06.py`, `lab06.task06_lanes`; helpers come from `solutions/cv_core.py`):

```python
def task06_lanes(image):
    edges,_=hough_segments(image); h,w=edges.shape
    roi=np.zeros_like(edges)
    polygon=np.int32([[(0,h-1),(int(w*.43),int(h*.55)),(int(w*.57),int(h*.55)),(w-1,h-1)]])
    cv.fillPoly(roi,polygon,255); cropped=cv.bitwise_and(edges,roi)
    raw=cv.HoughLinesP(cropped,1,np.pi/180,25,minLineLength=30,maxLineGap=25)
    groups={'left':[],'right':[]}; output=image.copy(); fits={}
    if raw is not None:
        for x1,y1,x2,y2 in raw.reshape(-1,4):
            if abs(x2-x1)<2: continue
            slope=(y2-y1)/(x2-x1)
            if not .35<abs(slope)<5: continue
            if slope<0 and (x1+x2)/2<w*.6: groups['left'] += [(x1,y1),(x2,y2)]
            elif slope>0 and (x1+x2)/2>w*.4: groups['right'] += [(x1,y1),(x2,y2)]
    for name,points in groups.items():
        if len(points)<2: continue
        p=np.array(points); a,b=np.polyfit(p[:,1],p[:,0],1) # x=a*y+b avoids division by slope
        ys=np.array([h-1,int(.6*h)]); xs=np.clip(a*ys+b,0,w-1).astype(int)
        cv.line(output,(int(xs[0]),int(ys[0])),(int(xs[1]),int(ys[1])),(0,255,0),5)
        fits[name]=list(zip(map(int,xs),map(int,ys)))
    return {'edges':edges,'ROI edges':cropped,'overlay':output,'lanes':fits}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Lane-line detection with Hough lines.

Task: You are working on an autonomous vehicle project, and one of the critical tasks is to detect lane
      markings on the road to ensure safe driving, using the Hough Line Transformation.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 5             # odd Gaussian kernel size; raise to 7-9 for noisy images
CANNY_LOW = 50             # hysteresis low threshold
CANNY_HIGH = 150           # hysteresis high threshold
ROI_TOP = 0.6              # top of the road trapezoid as a fraction of image height
HOUGH_THRESHOLD = 20       # minimum votes (accumulator threshold)
MIN_LINE_LENGTH = 20       # shortest segment kept, pixels
MAX_LINE_GAP = 60          # largest gap joined into one segment, pixels
MIN_SLOPE = 0.4            # ignore segments flatter than this |slope|


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

    # Step 5: Keep only the road region in front of the car
    h, w = edges.shape
    polygon = np.int32([[(0, h - 1), (int(w * 0.45), int(h * ROI_TOP)), (int(w * 0.55), int(h * ROI_TOP)), (w - 1, h - 1)]])
    roi = np.zeros_like(edges)
    cv.fillPoly(roi, polygon, 255)
    edges = cv.bitwise_and(edges, roi)
    stages['Edges in road ROI'] = edges

    # Step 6: Find line segments with probabilistic Hough
    found = cv.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                           minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    lines = np.empty((0, 4), int) if found is None else found.reshape(-1, 4)
    report['line_segments'] = len(lines)

    # Step 7: Split left/right lane segments and fit one line each
    h, w = edges.shape
    lanes = {}
    for side in ('left', 'right'):
        pts = []
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < MIN_SLOPE:
                continue                      # nearly horizontal: not a lane marking
            if (side == 'left') == (slope < 0):
                pts += [(x1, y1), (x2, y2)]
        if len(pts) >= 2:
            p = np.array(pts, float)
            a, b = np.polyfit(p[:, 1], p[:, 0], 1)          # x = a*y + b (stable for steep lines)
            ys = np.array([h - 1, int(h * ROI_TOP)])
            lanes[side] = [(int(a * yy + b), int(yy)) for yy in ys]
    report['lanes_found'] = sorted(lanes)

    # Step 8: Draw the lanes on the frame
    vis = img.copy()
    overlay = np.zeros_like(img)
    for (p1, p2) in lanes.values():
        cv.line(overlay, p1, p2, (0, 255, 0), 10)
    vis = cv.addWeighted(vis, 1.0, overlay, 0.8, 0)
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
