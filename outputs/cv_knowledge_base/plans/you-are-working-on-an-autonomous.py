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
