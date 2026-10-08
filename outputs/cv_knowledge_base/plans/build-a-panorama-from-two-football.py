"""Panorama stitching with SIFT and homography.

Task: Build a panorama from two football field images using point correspondences.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
RATIO = 0.75               # Lowe's ratio


def warp_pair(base, new, H):
    """Warp new into base's frame with H (new -> base) on a canvas that fits both."""
    hb, wb = base.shape[:2]; hn, wn = new.shape[:2]
    corners = np.float32([[0, 0], [wn, 0], [wn, hn], [0, hn]]).reshape(-1, 1, 2)
    moved = cv.perspectiveTransform(corners, H)
    allc = np.concatenate([moved, np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)])
    xmin, ymin = np.floor(allc.min(axis=(0, 1))).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=(0, 1))).astype(int)
    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], float)
    canvas = cv.warpPerspective(new, shift @ H, (xmax - xmin, ymax - ymin))
    region = canvas[-ymin:-ymin + hb, -xmin:-xmin + wb]
    keep = base.sum(axis=2) > 0
    region[keep] = base[keep]
    return canvas


def run(imgs):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Warp and blend the images into a panorama
    pano = imgs[0]
    for nxt in imgs[1:]:
        sift = cv.SIFT_create()
        k1, d1 = sift.detectAndCompute(cv.cvtColor(nxt, cv.COLOR_BGR2GRAY), None)
        k2, d2 = sift.detectAndCompute(cv.cvtColor(pano, cv.COLOR_BGR2GRAY), None)
        pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(d1, d2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < 4:
            raise ValueError('Not enough overlap to stitch')
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)        # maps nxt -> pano
        pano = warp_pair(pano, nxt, H)
        report.setdefault('matches_per_pair', []).append(len(good))
    vis = pano
    report['panorama_size'] = vis.shape[1::-1]
    stages['Panorama'] = vis
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
    paths = sys.argv[1:] or ['left.jpg', 'right.jpg']
    imgs = [cv.imread(p) for p in paths]
    if len(imgs) < 2 or any(i is None for i in imgs):
        raise FileNotFoundError('need at least two readable images')
    stages, report = run(imgs)
    for name, value in report.items():
        print(f'{name}: {value}')
    show(imgs[0], stages)
    cv.imwrite('panorama.png', stages['Panorama'])

if __name__ == '__main__':
    main()
