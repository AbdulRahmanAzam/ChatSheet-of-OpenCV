"""FROZEN custom coding tasks for the code generator (written before the generator; never tune on these).

Each task: the question a student would get, the function signature to write, a synthetic input with a
known answer, and a check of the returned value. None is answered by a planner recipe as-is: each needs
new code (a different output, a combination, or an extra condition).
"""
import cv2 as cv
import numpy as np

R = np.random.default_rng(3)


def _noisy(img, s=3):
    return np.clip(img.astype(np.int16) + R.normal(0, s, img.shape).astype(np.int16), 0, 255).astype(np.uint8)


def coins():
    img = np.full((400, 600, 3), 60, np.uint8)
    for (x, y), r in zip([(100, 100), (300, 110), (480, 120), (150, 300), (380, 290)], [30, 36, 45, 33, 40]):
        cv.circle(img, (x, y), r, (200, 200, 200), -1)
    return _noisy(img)


def blobs():
    img = np.zeros((300, 400, 3), np.uint8)
    cv.rectangle(img, (20, 20), (80, 70), (255, 255, 255), -1)        # 61*51 = 3111
    cv.rectangle(img, (150, 100), (290, 220), (255, 255, 255), -1)    # 141*121 = 17061
    cv.circle(img, (340, 60), 25, (255, 255, 255), -1)
    return img


def quadrants():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:100, :100], img[:100, 100:], img[100:, :100], img[100:, 100:] = 50, 100, 150, 200
    return img


def street():
    img = np.full((300, 500, 3), 110, np.uint8)
    cv.rectangle(img, (60, 150), (200, 230), (0, 220, 240), -1)       # yellow (BGR)
    cv.rectangle(img, (300, 200), (450, 260), (200, 80, 30), -1)      # blue
    return img


def frames():
    a = np.full((240, 320, 3), 90, np.uint8)
    b = a.copy(); cv.rectangle(b, (150, 140), (209, 219), (230, 230, 230), -1)
    return a, b


def lines_img():
    img = np.zeros((300, 400, 3), np.uint8)
    cv.line(img, (50, 50), (350, 50), (255, 255, 255), 3)      # 300 px horizontal
    cv.line(img, (60, 100), (60, 250), (255, 255, 255), 3)     # 150 px vertical
    return img


def rect_img():
    img = np.zeros((200, 200, 3), np.uint8)
    cv.rectangle(img, (50, 60), (150, 140), (255, 255, 255), -1)
    return img


def touching():
    img = np.zeros((300, 400, 3), np.uint8)
    for c in [(100, 150), (185, 150), (310, 150)]:
        cv.circle(img, c, 45, (255, 255, 255), -1)
    return img


def texture(h, w, seed):
    r = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(h * w // 900):
        color = tuple(int(c) for c in r.integers(0, 256, 3))
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        cv.circle(img, (x, y), int(r.integers(4, 25)), color, -1) if r.random() < .5 else \
            cv.rectangle(img, (x, y), (x + int(r.integers(5, 40)), y + int(r.integers(5, 40))), color, -1)
    return img


def scene_with_ref():
    ref = texture(150, 200, 1); scene = texture(480, 640, 2)
    scene[200:350, 300:500] = ref
    return ref, scene


def quad_img():
    img = np.zeros((400, 400, 3), np.uint8)
    pts = np.int32([[60, 40], [340, 70], [360, 330], [40, 300]])
    cv.fillPoly(img, [pts], (30, 140, 220))
    return img, pts.tolist()


def signal():
    x = np.sin(np.linspace(0, 20, 1000)) * 0.5 + R.normal(0, 0.05, 1000)
    x[[250, 600, 900]] += [4, -4, 4]
    return x


near = lambda a, b, tol: abs(float(a) - float(b)) <= tol

TASKS = [
    dict(id='coin_largest', sig='solve(img)',
         prompt='Write solve(img) that detects the coins in a BGR image with Hough circles and returns a dict with '
                '"count" (number of coins) and "largest_radius" (radius in pixels of the biggest coin).',
         make=lambda: (coins(),), check=lambda r: r['count'] == 5 and near(r['largest_radius'], 45, 4)),
    dict(id='largest_area', sig='solve(img)',
         prompt='Write solve(img) that thresholds the image, finds the white objects with contours and returns a dict '
                'with "count" and "largest_area" (contour area of the biggest object).',
         make=lambda: (blobs(),), check=lambda r: r['count'] == 3 and near(r['largest_area'], 140 * 120, 600)),
    dict(id='quadrant_means', sig='solve(img)',
         prompt='Write solve(img) that splits the grayscale version of the image into four equal quadrants and returns '
                'a list of their mean brightness in the order top-left, top-right, bottom-left, bottom-right.',
         make=lambda: (quadrants(),), check=lambda r: all(near(a, b, 1) for a, b in zip(r, [50, 100, 150, 200]))),
    dict(id='yellow_box', sig='solve(img)',
         prompt='Write solve(img) that finds the yellow object using the HSV color space and returns its bounding box '
                'as a tuple (x, y, w, h).',
         make=lambda: (street(),), check=lambda r: all(near(a, b, 3) for a, b in zip(r, (60, 150, 141, 81)))),
    dict(id='blue_fraction', sig='solve(img)',
         prompt='Write solve(img) that returns the percentage (0-100) of image pixels that are blue, using HSV.',
         make=lambda: (street(),), check=lambda r: near(r, 151 * 61 / 1500, 0.5)),
    dict(id='change_box', sig='solve(before, after)',
         prompt='Write solve(before, after) that compares two frames from a static camera and returns the bounding box '
                '(x, y, w, h) of the region that changed.',
         make=lambda: frames(), check=lambda r: all(near(a, b, 4) for a, b in zip(r, (150, 140, 60, 80)))),
    dict(id='longest_line', sig='solve(img)',
         prompt='Write solve(img) that uses Canny and HoughLinesP and returns the length in pixels of the longest '
                'detected line segment.',
         make=lambda: (lines_img(),), check=lambda r: near(r, 300, 15)),
    dict(id='hv_lines', sig='solve(img)',
         prompt='Write solve(img) that detects line segments with HoughLinesP and returns a dict with "horizontal" '
                'and "vertical" saying whether at least one horizontal and at least one vertical line was found (True/False).',
         make=lambda: (lines_img(),), check=lambda r: r['horizontal'] is True and r['vertical'] is True),
    dict(id='rotate_cw', sig='solve(img)',
         prompt='Write solve(img) that rotates the image 90 degrees clockwise without cropping and returns the rotated image.',
         make=lambda: (np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),),
         check=lambda r: r.shape == (3, 2, 3) and np.array_equal(r, np.rot90(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), -1))),
    dict(id='crop_largest', sig='solve(img)',
         prompt='Write solve(img) that crops the image to the bounding box of the largest white object and returns the crop.',
         make=lambda: (blobs(),), check=lambda r: abs(r.shape[0] - 121) <= 2 and abs(r.shape[1] - 141) <= 2),
    dict(id='otsu_white', sig='solve(img)',
         prompt='Write solve(img) that applies Otsu thresholding to the grayscale image and returns a dict with '
                '"threshold" (the value Otsu chose) and "white_pixels" (number of white pixels in the result).',
         make=lambda: (rect_img(),), check=lambda r: 0 <= r['threshold'] < 255 and r['white_pixels'] == 101 * 81),
    dict(id='watershed_count', sig='solve(img)',
         prompt='Write solve(img) that separates touching objects with marker-based watershed and returns the number '
                'of objects.',
         make=lambda: (touching(),), check=lambda r: r == 3),
    dict(id='corner_count', sig='solve(img)',
         prompt='Write solve(img) that finds the corners of the white rectangle with cv2.goodFeaturesToTrack and '
                'returns how many corners were found.',
         make=lambda: (rect_img(),), check=lambda r: r == 4),
    dict(id='hist_peak', sig='solve(img)',
         prompt='Write solve(img) that computes the grayscale histogram with cv2.calcHist and returns the most '
                'frequent gray level.',
         make=lambda: (quadrants()[:, :150],), check=lambda r: int(r) in (50, 150)),
    dict(id='spikes', sig='solve(signal)',
         prompt='Write solve(signal) that takes a 1-D numpy array of sensor readings and returns the sorted list of '
                'indices whose value is more than 3 standard deviations from the mean.',
         make=lambda: (signal(),), check=lambda r: list(r) == [250, 600, 900]),
    dict(id='warp_mean', sig='solve(img, corners)',
         prompt='Write solve(img, corners) where corners are four (x, y) points (top-left, top-right, bottom-right, '
                'bottom-left). Warp that region to a 200x300 (width x height) top-down view and return the warped image.',
         make=lambda: quad_img(), check=lambda r: r.shape[:2] == (300, 200) and np.allclose(r[20:-20, 20:-20].reshape(-1, 3).mean(0), (30, 140, 220), atol=6)),
    dict(id='kmeans_centers', sig='solve(img)',
         prompt='Write solve(img) that clusters the grayscale pixel values with cv2.kmeans into 4 clusters and '
                'returns the sorted list of cluster centers.',
         make=lambda: (quadrants(),), check=lambda r: all(near(a, b, 2) for a, b in zip(sorted(np.ravel(r)), [50, 100, 150, 200]))),
    dict(id='sift_found', sig='solve(ref, scene)',
         prompt='Write solve(ref, scene) that uses SIFT and the ratio test to decide whether the reference object '
                'appears in the scene; return a dict with "found" (True/False) and "good_matches" (count).',
         make=lambda: scene_with_ref(), check=lambda r: r['found'] is True and r['good_matches'] >= 10),
    dict(id='flip_compare', sig='solve(img)',
         prompt='Write solve(img) that flips the image horizontally and returns the number of pixels (rows*cols) whose '
                'grayscale value changed.',
         make=lambda: (quadrants(),), check=lambda r: r == 200 * 200),
    dict(id='blur_diff', sig='solve(img)',
         prompt='Write solve(img) that applies a 5x5 Gaussian blur to the grayscale image and returns the maximum '
                'absolute difference between the blurred and the original grayscale image.',
         make=lambda: (rect_img(),), check=lambda r: 100 <= r <= 200),
]
