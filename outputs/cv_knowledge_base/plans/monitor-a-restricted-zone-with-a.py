"""Restricted-zone change monitoring.

Task: Monitor a restricted zone with a camera and raise an alert when someone enters it.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
DIFF_THRESHOLD = 30        # gray-level change that counts as motion
MORPH_KSIZE = 5            # structuring element size
ZONE = [[100, 100], [300, 100], [300, 300], [100, 300]] # restricted zone polygon (x, y)
MIN_AREA = 200             # smallest changed area, pixels
PERSIST_FRAMES = 3         # frames in a row before alert


def run(img, state):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Compare each frame with the background
    g = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5, 5), 0)
    if 'background' not in state:
        state['background'] = g          # first frame (scene empty) becomes the reference
    diff = cv.absdiff(g, state['background'])
    _, mask = cv.threshold(diff, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
    mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    stages['Difference mask'] = mask

    # Step 3: Clean the mask with opening and closing
    kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # remove small specks
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # fill small holes
    stages['Cleaned mask'] = mask

    # Step 4: Raise an alert when change stays inside the restricted zone
    zone = np.zeros_like(mask)
    cv.fillPoly(zone, [np.int32(ZONE)], 255)
    inside = cv.bitwise_and(mask, zone)
    contours, _ = cv.findContours(inside, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
    state['streak'] = state.get('streak', 0) + 1 if contours else 0
    report['alert'] = state['streak'] >= PERSIST_FRAMES

    # Step 5: Draw zone, moving objects and alert
    vis = img.copy()
    cv.polylines(vis, [np.int32(ZONE)], True, (255, 0, 0), 2)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if report['alert']:
        cv.putText(vis, 'ALERT: zone entered', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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
    source = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
    cap = cv.VideoCapture(int(source) if source.isdigit() else source)   # 0 = webcam
    if not cap.isOpened():
        raise FileNotFoundError(source)
    fps = cap.get(cv.CAP_PROP_FPS) or 25
    state = {}
    writer, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stages, report = run(frame, state)
        out = stages.get('Result', frame)
        if writer is None:
            writer = cv.VideoWriter('result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, out.shape[1::-1])
        writer.write(out)
        n += 1
        if n % 25 == 1 or report.get('alert'):
            print(f'frame {n}:', {k: v for k, v in report.items() if not isinstance(v, (list, np.ndarray))})
    cap.release()
    if writer is not None:
        writer.release()
    print(f'{n} frames written to result.mp4')

if __name__ == '__main__':
    main()
