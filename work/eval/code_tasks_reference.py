"""Reference solutions that prove every frozen code task is solvable and its check is right (not shown to the generator)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2 as cv
import numpy as np
import code_tasks as C

g = lambda i: cv.cvtColor(i, cv.COLOR_BGR2GRAY)


def cnts(img):
    _, m = cv.threshold(g(img), 127, 255, cv.THRESH_BINARY)
    return cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]


def segs(img):
    return cv.HoughLinesP(cv.Canny(g(img), 50, 150), 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10).reshape(-1, 4)


def coin(img):
    c = cv.HoughCircles(cv.GaussianBlur(g(img), (9, 9), 2), cv.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=20, maxRadius=60)[0]
    return dict(count=len(c), largest_radius=float(c[:, 2].max()))


def watershed(img):
    _, m = cv.threshold(g(img), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    op = cv.morphologyEx(m, cv.MORPH_OPEN, k, iterations=2)
    bg = cv.dilate(op, k, iterations=3)
    d = cv.distanceTransform(op, cv.DIST_L2, 5)
    fg = np.uint8(d > 0.5 * d.max()) * 255
    n, mk = cv.connectedComponents(fg)
    mk = mk + 1; mk[cv.subtract(bg, fg) == 255] = 0
    mk = cv.watershed(img, mk)
    return len(set(np.unique(mk)) - {-1, 1})


def sift(ref, scene):
    s = cv.SIFT_create()
    _, a = s.detectAndCompute(g(ref), None); _, b = s.detectAndCompute(g(scene), None)
    good = [p[0] for p in cv.BFMatcher().knnMatch(a, b, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    return dict(found=len(good) >= 10, good_matches=len(good))


REF = {
    'coin_largest': coin,
    'largest_area': lambda img: dict(count=len(cnts(img)), largest_area=max(cv.contourArea(c) for c in cnts(img))),
    'quadrant_means': lambda img: [float(q.mean()) for q in (g(img)[:100, :100], g(img)[:100, 100:], g(img)[100:, :100], g(img)[100:, 100:])],
    'yellow_box': lambda img: cv.boundingRect(cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV), np.array([20, 100, 100]), np.array([35, 255, 255]))),
    'blue_fraction': lambda img: 100 * (cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV), np.array([90, 80, 50]), np.array([130, 255, 255])) > 0).mean(),
    'change_box': lambda a, b: cv.boundingRect(cv.threshold(cv.absdiff(g(a), g(b)), 30, 255, cv.THRESH_BINARY)[1]),
    'longest_line': lambda img: max(np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segs(img)),
    'hv_lines': lambda img: dict(horizontal=bool(any(abs(y2 - y1) < 5 for x1, y1, x2, y2 in segs(img))),
                                 vertical=bool(any(abs(x2 - x1) < 5 for x1, y1, x2, y2 in segs(img)))),
    'rotate_cw': lambda img: cv.rotate(img, cv.ROTATE_90_CLOCKWISE),
    'crop_largest': lambda img: (lambda c: img[c[1]:c[1] + c[3], c[0]:c[0] + c[2]])(cv.boundingRect(max(cnts(img), key=cv.contourArea))),
    'otsu_white': lambda img: (lambda t, m: dict(threshold=t, white_pixels=int((m > 0).sum())))(*cv.threshold(g(img), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)),
    'watershed_count': watershed,
    'corner_count': lambda img: len(cv.goodFeaturesToTrack(g(img), 10, 0.1, 10)),
    'hist_peak': lambda img: int(cv.calcHist([g(img)], [0], None, [256], [0, 256]).argmax()),
    'spikes': lambda s: sorted(np.flatnonzero(np.abs(s - s.mean()) > 3 * s.std()).tolist()),
    'warp_mean': lambda img, c: cv.warpPerspective(img, cv.getPerspectiveTransform(np.float32(c), np.float32([[0, 0], [199, 0], [199, 299], [0, 299]])), (200, 300)),
    'kmeans_centers': lambda img: sorted(cv.kmeans(g(img).reshape(-1, 1).astype(np.float32), 4, None, (3, 100, 0.2), 5, cv.KMEANS_PP_CENTERS)[2].ravel().tolist()),
    'sift_found': sift,
    'flip_compare': lambda img: int((g(cv.flip(img, 1)) != g(img)).sum()),
    'blur_diff': lambda img: int(np.abs(cv.GaussianBlur(g(img), (5, 5), 0).astype(int) - g(img).astype(int)).max()),
}

if __name__ == '__main__':
    ok = 0
    for t in C.TASKS:
        try:
            r = REF[t['id']](*t['make']())
            good = bool(t['check'](r))
        except Exception as e:
            good, r = False, f'{type(e).__name__}: {e}'
        ok += good
        print('OK ' if good else 'BAD', t['id'], '' if good else str(r)[:150])
    print(f'{ok}/{len(C.TASKS)} reference solutions pass')
