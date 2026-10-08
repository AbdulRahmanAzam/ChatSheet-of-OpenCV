"""Marker-based watershed segmentation.

Task: Implement the complete marker-based watershed pipeline.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
BLUR_KSIZE = 5             # odd Gaussian kernel size; raise to 7-9 for noisy images
THRESH_TYPE = cv.THRESH_BINARY # or cv.THRESH_BINARY_INV for dark objects
DIST_FRACTION = 0.5        # fraction of max distance for sure foreground


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Smooth noise with a Gaussian blur
    smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    stages['Blurred'] = smooth

    # Step 4: Otsu automatic threshold
    otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
    report['otsu_threshold'] = float(otsu_value)
    stages[f'Otsu (T={otsu_value:.0f})'] = mask

    # Step 5: Separate touching objects with marker-based watershed
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, DIST_FRACTION * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1          # background becomes 1, not 0
    markers[unknown == 255] = 0    # 0 = let watershed decide
    markers = cv.watershed(img, markers)
    report['count'] = len(set(np.unique(markers)) - {-1, 1})
    stages['Distance transform'] = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    stages['Sure foreground'] = sure_fg

    # Step 6: Draw watershed boundaries
    vis = img.copy()
    vis[markers == -1] = (0, 0, 255)
    cv.putText(vis, f"Objects: {report['count']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
