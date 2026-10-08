"""Image fusion of two registered images.

Task: Fuse a registered CT slice and MRI slice into one image.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
WEIGHT = 0.5               # weight of the first image (0-1)


def run(ref, img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Fuse the two registered images
    if ref.shape != img.shape:
        img = cv.resize(img, (ref.shape[1], ref.shape[0]))   # registered slices must share one size
    vis = cv.addWeighted(ref, WEIGHT, img, 1 - WEIGHT, 0)
    stages['Fused'] = vis
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
    ref = cv.imread(sys.argv[1] if len(sys.argv) > 1 else 'reference.jpg')
    img = cv.imread(sys.argv[2] if len(sys.argv) > 2 else 'scene.jpg')
    if ref is None or img is None:
        raise FileNotFoundError('check the reference and scene paths')
    stages, report = run(ref, img)
    for name, value in report.items():
        print(f'{name}: {value}')
    show(img, stages)
    cv.imwrite('result.png', stages['Result'])

if __name__ == '__main__':
    main()
