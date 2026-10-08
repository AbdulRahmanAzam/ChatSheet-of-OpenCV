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
