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
