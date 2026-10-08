"""Basic image operations.

Task: Threshold the image and then rotate it 45 degrees with scaling.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
THRESHOLD = 127            # cutoff 0-255
ANGLE = 45                 # degrees, positive = counter-clockwise
SCALE = 1.0                # scale factor


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Threshold the image
    _, mask = cv.threshold(gray, THRESHOLD, 255, cv.THRESH_BINARY)
    img = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)       # later steps work on the thresholded image
    stages['Threshold'] = mask

    # Step 4: Rotate (and scale) about the center
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), ANGLE, SCALE)
    warped = cv.warpAffine(img, M, (w, h))
    report['matrix'] = np.round(M, 3).tolist()
    stages[f'Rotated {ANGLE} deg'] = warped
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
