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
