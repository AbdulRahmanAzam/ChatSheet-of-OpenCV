"""Thresholding: global vs Otsu vs adaptive.

Task: Sweep the block size and C of adaptive thresholding on a document.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 5             # odd Gaussian kernel size; raise to 7-9 for noisy images
THRESHOLD = 127            # fixed cutoff 0-255
THRESH_TYPE = cv.THRESH_BINARY_INV # or cv.THRESH_BINARY_INV for dark objects
BLOCK_SIZE = 31            # odd neighbourhood size
C = 10                     # constant subtracted from the local mean


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Smooth noise with a Gaussian blur
    smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    stages['Blurred'] = smooth

    # Step 4: Global threshold
    _, mask = cv.threshold(smooth, THRESHOLD, 255, THRESH_TYPE)
    stages['Global threshold'] = mask

    # Step 5: Otsu automatic threshold
    otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
    report['otsu_threshold'] = float(otsu_value)
    stages[f'Otsu (T={otsu_value:.0f})'] = mask

    # Step 6: Adaptive (local) threshold
    mask = cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TYPE, BLOCK_SIZE, C)
    stages['Adaptive threshold'] = mask
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
