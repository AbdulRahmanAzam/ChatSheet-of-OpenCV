"""Object recognition with SIFT, ratio test and RANSAC homography.

Task: You are responsible for managing computer assets in a busy computing lab. Implement a computer
      vision solution using the Scale-Invariant Feature Transform (SIFT) to automatically recognize
      and identify individual computer systems and their unique components, such as monitors and
      keyboards, to maintain an up-to-date inventory.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
RATIO = 0.75               # Lowe's ratio; lower = stricter
RANSAC_PX = 5.0            # reprojection error allowed, pixels
MIN_INLIERS = 10           # inliers needed to accept


def run(ref, img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Detect SIFT keypoints and descriptors in both images
    sift = cv.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(cv.cvtColor(ref, cv.COLOR_BGR2GRAY), None)
    kp, des = sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)
    report['keypoints_ref'] = len(kp_ref)
    report['keypoints_scene'] = len(kp)

    # Step 3: Match descriptors with Lowe's ratio test
    good = []
    if des_ref is not None and des is not None and len(des) >= 2:
        for pair in cv.BFMatcher(cv.NORM_L2).knnMatch(des_ref, des, k=2):
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                good.append(pair[0])
    report['good_matches'] = len(good)

    # Step 4: Estimate the homography with RANSAC
    H, inliers = None, 0
    if len(good) >= 4:
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inlier_mask = cv.findHomography(src, dst, cv.RANSAC, RANSAC_PX)
        inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
    report['inliers'] = inliers
    report['found'] = H is not None and inliers >= MIN_INLIERS

    # Step 5: Draw the recognised object outline
    vis = img.copy()
    if report['found']:
        h, w = ref.shape[:2]
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        box = cv.perspectiveTransform(corners, H)
        cv.polylines(vis, [np.int32(box)], True, (0, 255, 0), 3)
        report['box'] = np.int32(box).reshape(-1, 2).tolist()
    cv.putText(vis, 'FOUND' if report['found'] else 'NOT FOUND', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
               (0, 255, 0) if report['found'] else (0, 0, 255), 2)
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
