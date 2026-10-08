"""Execute the planner's generated scripts on synthetic inputs with known answers.

For every case: plan the task text, generate the script, import it, call run(...) on a synthetic
image/video/signal whose ground truth we drew ourselves, and check the report. A plan only counts
as correct if its own code produces the right answer.

    python work/plan_selftest.py            # all cases
    python work/plan_selftest.py screens    # one case
"""
from pathlib import Path
import sys, types, traceback

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner  # noqa: E402

RNG = np.random.default_rng(7)


def noisy(img, sigma=4):
    return np.clip(img.astype(np.int16) + RNG.normal(0, sigma, img.shape).astype(np.int16), 0, 255).astype(np.uint8)


def texture(h, w, seed):
    """Feature-rich synthetic picture (random shapes) for SIFT matching and stitching."""
    r = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(h * w // 900):
        color = tuple(int(c) for c in r.integers(0, 256, 3))
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        if r.random() < 0.5:
            cv.circle(img, (x, y), int(r.integers(4, 25)), color, -1)
        else:
            cv.rectangle(img, (x, y), (x + int(r.integers(5, 40)), y + int(r.integers(5, 40))), color, -1)
    return img


# ---------------------------------------------------------------- synthetic inputs
def lab_screens():
    img = np.full((560, 900, 3), 115, np.uint8)
    states = {(0, 0): 'ON', (0, 1): 'OFF', (0, 2): 'ON', (1, 0): 'OFF', (1, 1): 'ON'}   # (1, 2) missing
    for (r, c), s in states.items():
        x, y = 60 + c * 280, 50 + r * 260
        cv.rectangle(img, (x, y), (x + 210, y + 140), (40, 40, 40), -1)                 # bezel
        cv.rectangle(img, (x + 10, y + 10), (x + 200, y + 130), (225, 225, 225) if s == 'ON' else (18, 18, 18), -1)
        cv.rectangle(img, (x + 30, y + 170), (x + 180, y + 190), (70, 70, 70), -1)      # keyboard (too thin to be a screen)
    return noisy(img), states


def road():
    img = np.full((360, 640, 3), 70, np.uint8)
    img[:150] = (200, 170, 120)                                                         # sky
    cv.line(img, (90, 359), (300, 200), (255, 255, 255), 6)
    for y0 in range(359, 200, -40):                                                     # dashed right lane
        t0, t1 = (359 - y0) / 159, min((359 - y0 + 25) / 159, 1)
        cv.line(img, (int(560 - 220 * t0), y0), (int(560 - 220 * t1), int(359 - 159 * t1)), (255, 255, 255), 6)
    return noisy(img)


def coins(n=6):
    img = np.full((480, 640, 3), 60, np.uint8)
    centers = [(100, 120), (260, 110), (450, 140), (140, 330), (330, 320), (520, 360)][:n]
    for i, (x, y) in enumerate(centers):
        cv.circle(img, (x, y), 32 + 3 * (i % 3), (190, 190, 200), -1)
        cv.circle(img, (x, y), 12, (150, 150, 160), 2)                                  # embossed detail
    return noisy(img), n


def pills(n=7):
    img = np.zeros((400, 600, 3), np.uint8)
    for i in range(n):
        cv.ellipse(img, (70 + (i % 4) * 140, 100 + (i // 4) * 200), (40, 25), 30 * i, 0, 360, (230, 230, 230), -1)
    return noisy(img, 6), n


def touching():
    img = np.zeros((300, 400, 3), np.uint8)
    for c in [(100, 150), (185, 150), (310, 150)]:                                    # two touching + one apart
        cv.circle(img, c, 45, (255, 255, 255), -1)
    return img, 3


def three_colors():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, :100] = (200, 40, 40); img[:, 100:200] = (40, 200, 40); img[:, 200:] = (40, 40, 200)
    return noisy(img, 2)


def street():
    img = np.full((300, 500, 3), 110, np.uint8)
    cv.rectangle(img, (60, 150), (200, 230), (0, 220, 240), -1)                         # yellow car (BGR)
    cv.circle(img, (350, 120), 40, (30, 30, 220), -1)                                   # red sign
    cv.rectangle(img, (300, 200), (450, 260), (200, 80, 30), -1)                        # blue car
    return noisy(img, 3), (60, 150, 141, 81)


def document():
    h, w = 300, 400
    ramp = np.tile(np.linspace(220, 90, w), (h, 1))                                     # uneven lighting
    img = np.dstack([ramp] * 3).astype(np.uint8)
    for i, y in enumerate(range(40, 280, 40)):
        cv.putText(img, 'line of text %d' % i, (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return noisy(img, 3)


def zone_frames():
    frames = []
    for t in range(12):
        f = np.full((240, 320, 3), 90, np.uint8)
        if t >= 5:
            cv.rectangle(f, (150 + t, 150), (210 + t, 230), (230, 230, 230), -1)        # intruder inside zone
        frames.append(noisy(f, 2))
    return frames


def sensor():
    t = np.arange(1024)
    x = np.sin(2 * np.pi * t / 128) + RNG.normal(0, 0.1, len(t))
    spikes = [300, 701]
    x[spikes] += [3.0, -2.5]
    return x, spikes


def run_plan(text):
    p = planner.build(text)
    mod = types.ModuleType('generated')
    exec(compile(planner.to_script(p), '<generated>', 'exec'), mod.__dict__)
    return p, mod


# ---------------------------------------------------------------- cases: (name, task text, check(plan, module) -> message or None)
def c_screens(p, m):
    img, truth = lab_screens()
    stages, rep = m.run(img)
    assert rep['screens_found'] == 5, rep
    assert rep['on'] == 3 and rep['off'] == 2, rep
    assert rep['missing'] == 1 and rep['missing_positions'][0][:2] == (1, 2), rep['missing_positions']


def c_lanes(p, m):
    stages, rep = m.run(road())
    assert rep['lanes_found'] == ['left', 'right'], rep


def c_circles(p, m):
    img, n = coins()
    stages, rep = m.run(img)
    assert rep['count'] == n, rep


def c_recognize(p, m):
    ref = texture(150, 200, 1)
    scene = texture(480, 640, 2)
    M = cv.getRotationMatrix2D((100, 75), 20, 0.8); M[:, 2] += (300, 200)
    obj = cv.warpAffine(ref, M, (640, 480)); mask = cv.warpAffine(np.full((150, 200), 255, np.uint8), M, (640, 480))
    scene[mask > 0] = obj[mask > 0]
    stages, rep = m.run(ref, scene)
    assert rep['found'] and rep['inliers'] >= m.MIN_INLIERS, rep
    center = np.mean(rep['box'], axis=0); truth = M @ np.array([100, 75, 1])
    assert np.linalg.norm(center - truth) < 10, (center, truth)
    stages, rep = m.run(ref, texture(480, 640, 3))                                       # object absent
    assert not rep['found'], rep


def c_panorama(p, m):
    full = texture(300, 800, 4)
    stages, rep = m.run([full[:, :480].copy(), full[:, 320:].copy()])
    w, h = rep['panorama_size']
    assert abs(w - 800) <= 15 and abs(h - 300) <= 15, rep


def c_count(p, m):
    img, n = pills()
    stages, rep = m.run(img)
    assert rep['count'] == n, rep


def c_watershed(p, m):
    img, n = touching()
    stages, rep = m.run(img)
    assert rep['count'] == n, rep


def c_kmeans(p, m):
    stages, rep = m.run(three_colors())
    assert rep['colors_k2'] == 2 and rep['colors_k4'] <= 4, rep


def c_color(p, m):
    img, (x, y, w, h) = street()
    stages, rep = m.run(img)
    assert rep['count'] == 1 and abs(rep['areas'][0] - w * h) < 0.1 * w * h, rep


def c_threshold(p, m):
    img = document()
    stages, rep = m.run(img)
    names = list(stages)
    assert any('Adaptive' in k for k in names) and any('Otsu' in k for k in names), names
    text_rows = slice(25, 45)                                                           # first text line
    adaptive = stages['Adaptive threshold']
    assert adaptive[text_rows, 20:300].mean() > 20, 'adaptive threshold lost the text'


def c_edges(p, m):
    stages, rep = m.run(pills()[0])
    assert stages['Canny edges'].max() == 255


def c_region(p, m):
    img = np.full((200, 200, 3), 50, np.uint8); img[40:160, 40:160] = 120; img[80:120, 80:120] = 135
    m.SEEDS = [(100, 100)]
    stages, rep = m.run(img)
    a = [rep[f'area_tol{t}'] for t in m.TOLERANCES]
    assert a == [40 * 40, 120 * 120, 120 * 120], a      # 30 still stops at the 50-level background


def c_zone(p, m):
    m.ZONE = [[120, 120], [300, 120], [300, 239], [120, 239]]
    state, alerts = {}, []
    for f in zone_frames():
        alerts.append(m.run(f, state)[1]['alert'])
    assert not any(alerts[:7]) and all(alerts[8:]), alerts


def c_enhance(p, m):
    img = np.dstack([np.clip(RNG.normal(120, 8, (200, 200)), 0, 255).astype(np.uint8)] * 3)
    stages, rep = m.run(img)
    assert rep['contrast_after'] > 2 * rep['contrast_before'], rep


def c_rotate(p, m):
    stages, rep = m.run(texture(200, 300, 5))
    M = np.array(rep['matrix'])
    assert abs(M[0, 0] - np.cos(np.radians(m.ANGLE))) < 1e-3, M


def c_perspective(p, m):
    img = texture(400, 400, 6)
    stages, rep = m.run(img)
    assert stages['Top-down view'].shape[:2] == (m.OUT_H, m.OUT_W)


def c_texture(p, m):
    stages, rep = m.run(texture(200, 200, 8))
    assert rep['hog_length'] == 3780 and rep['lbp_bins'] == 256, rep


def c_corners(p, m):
    img = np.zeros((200, 200, 3), np.uint8); cv.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    stages, rep = m.run(img)
    assert rep['corners'] > 0, rep


def c_wavelet(p, m):
    x, spikes = sensor()
    stages, rep, denoised, anomalies = m.run(x)
    assert set(spikes) <= set(rep['anomalies']) and len(rep['anomalies']) <= 6, rep


def c_basics(p, m):
    img = texture(400, 600, 9)
    stages, rep = m.run(img)
    assert stages['Center ROI'].shape == (200, 200, 3) and 'Blurred' in stages, list(stages)


CASES = [
    ('screens', 'You are working on a computer vision project for monitoring computer lab usage and ensuring the availability '
     'of computer screens. Your goal is to implement screen detection using the Hough Line Transformation. The lab contains '
     'rows of computers, and you need to identify the boundaries of computer screens to monitor their status (on or off) '
     'and detect any anomalies, such as missing screens.', c_screens),
    ('lanes', 'You are working on an autonomous vehicle project, and one of the critical tasks is to detect lane markings on '
     'the road to ensure safe driving. Use the Hough Line Transformation.', c_lanes),
    ('circles', 'Count the coins in an image using the Hough circle transform.', c_circles),
    ('recognize', 'You have a reference image of the object you want to recognize and a set of test images. Identify and '
     'locate the object in each test image, even if it appears at different scales, orientations, or under partial '
     'occlusion, using SIFT, and draw bounding boxes.', c_recognize),
    ('panorama', 'Create a panoramic image by stitching multiple overlapping images captured while panning, using SIFT to '
     'match key points between adjacent images.', c_panorama),
    ('count', 'Count the number of pills on the tray and draw a box around each one.', c_count),
    ('watershed', 'Segment touching coins and count them using marker-based watershed.', c_watershed),
    ('kmeans', 'Segment the image with k-means clustering for K=2, 4 and 6 and compare the results.', c_kmeans),
    ('color', 'Isolate the yellow car in the street image using the HSV color space.', c_color),
    ('threshold', 'Segment a scanned document with uneven illumination using global and adaptive thresholding and compare them.', c_threshold),
    ('edges', 'Apply Canny edge detection to the image.', c_edges),
    ('region', 'Implement region growing from a seed point with tolerances of 5, 15 and 30 and compare the regions.', c_region),
    ('zone', 'Monitor a restricted zone in a security camera video and raise an alert when someone enters it.', c_zone),
    ('enhance', 'Enhance a low-contrast chest X-ray so the bones are visible.', c_enhance),
    ('rotate', 'Rotate the satellite image by 30 degrees around its center.', c_rotate),
    ('perspective', 'Get a top-down view of a tilted square painting from its four corners using a perspective transform.', c_perspective),
    ('texture', 'Analyse material texture using HOG and LBP features.', c_texture),
    ('corners', 'Detect corners in the image with the Harris detector.', c_corners),
    ('wavelet', 'Detect anomalies in industrial sensor data using the wavelet transformation.', c_wavelet),
    ('basics', 'Apply a 25x25 blur to the image and crop the exact center ROI.', c_basics),
]


def main(only=None):
    ok = 0
    cases = [c for c in CASES if only is None or c[0] == only]
    for name, text, check in cases:
        try:
            p, m = run_plan(text)
            check(p, m)
            ok += 1
            print(f'PASS {name:12s} -> {p["recipe"]} ({len(p["steps"])} steps)')
        except Exception as e:
            print(f'FAIL {name:12s} -> {type(e).__name__}: {str(e)[:300]}')
            if only:
                traceback.print_exc()
    print(f'\n{ok}/{len(cases)} plans produced the right answer on synthetic ground truth')
    return ok == len(cases)


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
