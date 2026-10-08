"""Texture and shape features (HOG and LBP).

Task: Analyse material texture using HOG and LBP.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----



def lbp_image(g):
    """8-neighbour, radius-1 local binary pattern codes (border pixels skipped)."""
    g = g.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(shifts):
        n = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((n >= c).astype(np.uint8) << bit)
    return code


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Compute Sobel gradients, magnitude and direction
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.degrees(np.arctan2(gy, gx))
    stages['Gradient magnitude'] = cv.convertScaleAbs(magnitude)

    # Step 4: HOG descriptor (shape / edge-direction features)
    patch = cv.resize(gray, (64, 128))
    hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_vec = hog.compute(patch).ravel()
    report['hog_length'] = len(hog_vec)

    # Step 5: LBP histogram (texture features)
    lbp = lbp_image(gray)
    lbp_hist = np.bincount(lbp.ravel(), minlength=256).astype(float)
    lbp_hist /= lbp_hist.sum()
    stages['LBP'] = lbp
    report['lbp_bins'] = len(lbp_hist)
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
