"""Step library for the offline planner.

Each step is one small, explainable unit of a computer-vision solution:
  needs   variables that must exist before the step runs
  makes   variables the step creates (or overwrites)
  params  tunable constants {NAME: (default, meaning)}; they become the script's PARAMETERS block
  code    Python lines that run inside run(...); they use the variables and params by name
  helpers module-level functions the code calls (added once to the script)
  why     why the step is there and how to choose its parameters
  topics  knowledge-base topic ids (pitfalls and manual references come from them)

Variable conventions: img (BGR uint8), ref (reference BGR), gray, smooth, edges, mask,
lines (N x 4), circles (N x 3), contours, vis (annotated BGR), stages (images to show),
report (numbers and verdicts to print), state (memory kept between video frames).
"""

S = {}


def step(id, title, needs, makes, code, why, topics=(), params=None, helpers='', watch=''):
    S[id] = dict(id=id, title=title, needs=list(needs), makes=list(makes), code=code.strip('\n'),
                 why=why, topics=list(topics), params=params or {}, helpers=helpers.strip('\n'), watch=watch)


# ---------------------------------------------------------------- preprocessing
step('gray', 'Convert to grayscale', ['img'], ['gray'], """
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
stages['Grayscale'] = gray
""", 'Edge, line, circle and threshold operations work on one intensity channel. OpenCV loads color as BGR, '
     'so the code must be COLOR_BGR2GRAY (not RGB2GRAY).', ['colors'],
     watch='cv.imread gives BGR order; using COLOR_RGB2GRAY swaps the red and blue weights.')

step('gaussian', 'Smooth noise with a Gaussian blur', ['gray'], ['smooth'], """
smooth = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
stages['Blurred'] = smooth
""", 'Noise creates false edges and false Hough votes. A Gaussian blur removes fine noise while keeping strong '
     'boundaries. Kernel size must be odd; bigger = smoother but weaker thin edges.', ['gaussian'],
     {'BLUR_KSIZE': (5, 'odd Gaussian kernel size; raise to 7-9 for noisy images')})

step('median', 'Remove speckle noise with a median blur', ['gray'], ['smooth'], """
smooth = cv.medianBlur(gray, MEDIAN_KSIZE)
stages['Median blurred'] = smooth
""", 'A median blur removes salt-and-pepper noise and keeps edges sharp; HoughCircles is very sensitive to noise, '
     'so the manual blurs before it.', ['median'],
     {'MEDIAN_KSIZE': (5, 'odd median kernel size')})

step('hsv', 'Convert to HSV color space', ['img'], ['hsv'], """
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
""", 'Hue separates "which color" from brightness, so one hue range keeps working in shade and light. '
     'OpenCV hue runs 0-179 (not 0-359).', ['color_spaces'])

# ---------------------------------------------------------------- enhancement (Lab 02)
step('equalize', 'Global histogram equalization', ['gray'], ['equalized'], """
equalized = cv.equalizeHist(gray)
stages['Equalized'] = equalized
""", 'Spreads the most frequent intensities over 0-255 to raise global contrast of a dull image.', ['equalization'])

step('clahe', 'Local contrast enhancement (CLAHE)', ['gray'], ['enhanced'], """
clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
enhanced = clahe.apply(gray)
stages['CLAHE'] = enhanced
""", 'CLAHE equalizes each tile separately and clips the histogram, so it brings out local detail (bones, tissue, '
     'text) without blowing out noise the way global equalization can.', ['equalization'],
     {'CLAHE_CLIP': (2.0, 'contrast limit; higher = stronger, noisier'), 'CLAHE_TILES': (8, 'tiles per side')})

step('gamma', 'Gamma correction', ['gray'], ['gamma_img'], """
table = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)]).astype(np.uint8)
gamma_img = cv.LUT(gray, table)
stages[f'Gamma {GAMMA}'] = gamma_img
""", 'Output = 255*(I/255)^gamma. gamma < 1 brightens dark regions, gamma > 1 darkens bright regions. '
     'A lookup table applies it to every pixel fast.', ['gamma'],
     {'GAMMA': (0.6, '<1 brightens shadows, >1 darkens')})

step('log', 'Log transform', ['gray'], ['log_img'], """
c = 255 / np.log1p(max(int(gray.max()), 1))
log_img = np.uint8(np.clip(c * np.log1p(gray.astype(np.float64)), 0, 255))
stages['Log transform'] = log_img
""", 's = c*log(1+r) expands dark values and compresses bright ones; c scales the brightest input to 255.', ['gamma'])

step('pseudocolor', 'Pseudocolor for visual inspection', ['enhanced'], ['colored'], """
colored = cv.applyColorMap(enhanced, COLORMAP)
stages['Pseudocolor'] = colored
""", 'The eye separates hues better than gray levels; a colormap makes intensity differences easy to see. '
     'It is a display aid, not new information.', ['pseudocolor'],
     {'COLORMAP': ('cv.COLORMAP_JET', 'any cv.COLORMAP_* constant')})

step('enhance_report', 'Measure the contrast gain', ['gray', 'enhanced'], [], """
report['contrast_before'] = round(float(gray.std()), 2)
report['contrast_after'] = round(float(enhanced.std()), 2)
""", 'The standard deviation of intensity is a simple contrast number; it should go up after enhancement.',
     ['histogram'])

# ---------------------------------------------------------------- edges and lines
step('canny', 'Detect edges with Canny', ['smooth'], ['edges'], """
edges = cv.Canny(smooth, CANNY_LOW, CANNY_HIGH)
stages['Canny edges'] = edges
""", 'Canny gives thin, connected edges (gradient, non-maximum suppression, hysteresis). Pixels above CANNY_HIGH are '
     'sure edges; pixels between the two thresholds survive only if connected to a sure edge. A 1:2 to 1:3 ratio is '
     'the usual choice. Raise both if texture gives too many edges.', ['canny'],
     {'CANNY_LOW': (50, 'hysteresis low threshold'), 'CANNY_HIGH': (150, 'hysteresis high threshold')})

step('lane_roi', 'Keep only the road region in front of the car', ['edges'], ['edges'], """
h, w = edges.shape
polygon = np.int32([[(0, h - 1), (int(w * 0.45), int(h * ROI_TOP)), (int(w * 0.55), int(h * ROI_TOP)), (w - 1, h - 1)]])
roi = np.zeros_like(edges)
cv.fillPoly(roi, polygon, 255)
edges = cv.bitwise_and(edges, roi)
stages['Edges in road ROI'] = edges
""", 'Lane markings sit in a trapezoid in front of the car; masking everything else removes trees, sky and other '
     'cars before voting.', ['mask', 'hough'],
     {'ROI_TOP': (0.6, 'top of the road trapezoid as a fraction of image height')})

step('hough_p', 'Find line segments with probabilistic Hough', ['edges'], ['lines'], """
found = cv.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                       minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
lines = np.empty((0, 4), int) if found is None else found.reshape(-1, 4)
report['line_segments'] = len(lines)
""", 'Each edge pixel votes for all lines (rho, theta) through it; cells with at least HOUGH_THRESHOLD votes are '
     'lines. HoughLinesP returns finite segments (x1,y1,x2,y2), which is what boundaries need. minLineLength drops '
     'short texture lines, maxLineGap joins broken edges. It returns None when nothing is found.', ['hough'],
     {'HOUGH_THRESHOLD': (50, 'minimum votes (accumulator threshold)'),
      'MIN_LINE_LENGTH': (40, 'shortest segment kept, pixels'),
      'MAX_LINE_GAP': (10, 'largest gap joined into one segment, pixels')})

step('hv_filter', 'Keep horizontal and vertical lines only', ['lines'], ['lines'], """
angle = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180
horizontal = np.minimum(angle, 180 - angle) < ANGLE_TOLERANCE
vertical = np.abs(angle - 90) < ANGLE_TOLERANCE
lines = lines[horizontal | vertical]
report['horizontal_lines'] = int(horizontal.sum())
report['vertical_lines'] = int(vertical.sum())
""", 'Screen borders are horizontal and vertical in a front view; dropping slanted lines removes keyboards, cables '
     'and chair edges.', ['hough', 'screen'],
     {'ANGLE_TOLERANCE': (10, 'degrees a line may lean and still count as horizontal/vertical')})

step('line_canvas', 'Draw the Hough lines on a blank canvas', ['lines', 'edges'], ['canvas'], """
canvas = np.zeros_like(edges)
for x1, y1, x2, y2 in lines:
    cv.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
canvas = cv.morphologyEx(canvas, cv.MORPH_CLOSE, np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8))
stages['Hough lines'] = canvas
""", 'Rebuilding the image from lines only keeps the straight structure and drops curved clutter; closing '
     'bridges small gaps where two segments almost meet, so the sides of one screen join into one shape.',
     ['hough', 'morphology'], {'CLOSE_KSIZE': (9, 'closing kernel that bridges gaps between segments')})

step('rects_from_canvas', 'Turn line outlines into screen rectangles', ['canvas'], ['screens'], """
contours, _ = cv.findContours(canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
screens = []
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    if w * h < MIN_SCREEN_AREA or not MIN_ASPECT <= w / h <= MAX_ASPECT:
        continue
    sides = side_support(canvas, (x, y, w, h))           # share of each side covered by Hough lines
    if sum(s >= SIDE_SUPPORT for s in sides) >= 3:       # one side may be hidden (person, cable, glare)
        screens.append((x, y, w, h))
screens = sort_rows(drop_nested(screens))
report['screens_found'] = len(screens)
""", 'A screen is a rectangle of the right size and shape whose sides are made of Hough lines. For each outline '
     'the bounding box is tested: at least 3 of its 4 sides must be covered by line pixels (SIDE_SUPPORT). The '
     'area and width/height filter rejects keyboards, desks and posters; boxes inside a bigger box (the inner '
     'screen edge inside the bezel) are dropped so each monitor is counted once.',
     ['contours', 'screen'],
     {'MIN_SCREEN_AREA': (2000, 'smallest screen area in pixels'),
      'MIN_ASPECT': (1.0, 'smallest width/height'), 'MAX_ASPECT': (2.5, 'largest width/height'),
      'SIDE_SUPPORT': (0.6, 'share of a side that must lie on Hough lines')},
     helpers='''
def side_support(canvas, box, band=4):
    """Fraction of each side (top, bottom, left, right) of box that has line pixels within band px."""
    x, y, w, h = box
    near = cv.dilate(canvas, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0
    y2, x2 = min(y + h - 1, canvas.shape[0] - 1), min(x + w - 1, canvas.shape[1] - 1)
    return [near[y, x:x2 + 1].mean(), near[y2, x:x2 + 1].mean(), near[y:y2 + 1, x].mean(), near[y:y2 + 1, x2].mean()]


def drop_nested(boxes):
    """Remove boxes that lie inside another (bigger) box."""
    inside = lambda a, b: a != b and a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]
    return [a for a in boxes if not any(inside(a, b) for b in boxes)]


def sort_rows(boxes):
    """Order boxes row by row (top to bottom), left to right inside a row."""
    if not boxes:
        return []
    row_height = np.median([b[3] for b in boxes])
    return sorted(boxes, key=lambda b: (int(round(b[1] / row_height)), b[0]))
''')

step('screen_state', 'Decide ON / OFF from brightness inside each screen', ['screens', 'gray'], ['states'], """
states = []
for x, y, w, h in screens:
    inner = gray[y + h // 5: y + h - h // 5, x + w // 5: x + w - w // 5]   # skip the bezel
    level = float(inner.mean())
    states.append({'box': (x, y, w, h), 'brightness': round(level, 1),
                   'state': 'ON' if level > ON_LEVEL else 'OFF'})
report['on'] = sum(s['state'] == 'ON' for s in states)
report['off'] = sum(s['state'] == 'OFF' for s in states)
""", 'A lit screen is much brighter than a dark one. The center of the box (inset 20%) is measured so the bezel '
     'does not count. Choose ON_LEVEL between the two groups seen in your image (look at the printed brightness).',
     ['screen', 'histogram'],
     {'ON_LEVEL': (100, 'mean gray level above which a screen counts as ON')})

step('missing_slots', 'Find missing screens from the row/column layout', ['screens'], ['missing'], """
missing = find_missing(screens, EXPECTED_SCREENS)
report['missing'] = len(missing)
report['missing_positions'] = missing
""", 'Computers stand in rows and columns. The centers of the found screens give the row and column positions; '
     'any (row, column) cell with no screen is reported missing. If you know the expected count, set '
     'EXPECTED_SCREENS so a whole missing row/column is also caught.', ['screen'],
     {'EXPECTED_SCREENS': (None, 'expected number of screens, or None to infer from the layout')},
     helpers='''
def cluster_1d(values, gap):
    """Group sorted numbers whose neighbours are closer than gap; return the group means."""
    groups = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] < gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def find_missing(boxes, expected=None):
    """Grid cells (row, col, x, y) that have no box. Assumes screens stand in rows and columns."""
    if not boxes:
        return []
    w = np.median([b[2] for b in boxes]); h = np.median([b[3] for b in boxes])
    centers = [(x + bw / 2, y + bh / 2) for x, y, bw, bh in boxes]
    cols = cluster_1d([c[0] for c in centers], w / 2)
    rows = cluster_1d([c[1] for c in centers], h / 2)
    missing = []
    for r, cy in enumerate(rows):
        for c, cx in enumerate(cols):
            if not any(abs(px - cx) < w / 2 and abs(py - cy) < h / 2 for px, py in centers):
                missing.append((r, c, int(cx), int(cy)))
    if expected is not None and len(boxes) + len(missing) < expected:
        missing.append(('unknown', 'unknown', expected - len(boxes) - len(missing), 'not on the grid'))
    return missing
''')

step('draw_screens', 'Draw screens, states and missing slots', ['img', 'states', 'missing'], ['vis'], """
vis = img.copy()
for s in states:
    x, y, w, h = s['box']
    color = (0, 255, 0) if s['state'] == 'ON' else (255, 128, 0)
    cv.rectangle(vis, (x, y), (x + w, y + h), color, 3)
    cv.putText(vis, s['state'], (x + 5, y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
for m in missing:
    if isinstance(m[2], int) and isinstance(m[3], int):
        cv.drawMarker(vis, (m[2], m[3]), (0, 0, 255), cv.MARKER_TILTED_CROSS, 60, 4)
        cv.putText(vis, 'MISSING', (m[2] - 50, m[3] + 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
stages['Result'] = vis
""", 'The annotated image is the deliverable: green = ON, blue = OFF, red cross = missing screen.', ['drawing'])

step('lane_fit', 'Split left/right lane segments and fit one line each', ['lines', 'edges'], ['lanes'], """
h, w = edges.shape
lanes = {}
for side in ('left', 'right'):
    pts = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE:
            continue                      # nearly horizontal: not a lane marking
        if (side == 'left') == (slope < 0):
            pts += [(x1, y1), (x2, y2)]
    if len(pts) >= 2:
        p = np.array(pts, float)
        a, b = np.polyfit(p[:, 1], p[:, 0], 1)          # x = a*y + b (stable for steep lines)
        ys = np.array([h - 1, int(h * ROI_TOP)])
        lanes[side] = [(int(a * yy + b), int(yy)) for yy in ys]
report['lanes_found'] = sorted(lanes)
""", 'In image coordinates (y grows downward) the left lane has negative slope and the right lane positive slope. '
     'Fitting x = a*y + b to all points of one side averages the dashed pieces into one stable lane line.',
     ['hough'], {'MIN_SLOPE': (0.4, 'ignore segments flatter than this |slope|')})

step('draw_lanes', 'Draw the lanes on the frame', ['img', 'lanes'], ['vis'], """
vis = img.copy()
overlay = np.zeros_like(img)
for (p1, p2) in lanes.values():
    cv.line(overlay, p1, p2, (0, 255, 0), 10)
vis = cv.addWeighted(vis, 1.0, overlay, 0.8, 0)
stages['Result'] = vis
""", 'Lines are drawn on a separate layer and blended so the road stays visible under them.', ['drawing', 'blend'])

step('draw_lines', 'Draw the detected line segments', ['img', 'lines'], ['vis'], """
vis = img.copy()
for x1, y1, x2, y2 in lines:
    cv.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
stages['Result'] = vis
""", 'Drawing the segments over the original lets you check every detection by eye.', ['drawing'])

# ---------------------------------------------------------------- circles
step('hough_circles', 'Find circles with the Hough gradient method', ['smooth'], ['circles'], """
found = cv.HoughCircles(smooth, cv.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
                        param1=CANNY_HIGH, param2=ACC_THRESHOLD, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
circles = np.empty((0, 3), int) if found is None else np.round(found[0]).astype(int)
report['count'] = len(circles)
""", 'HOUGH_GRADIENT runs Canny internally (param1 = its high threshold) and votes for centers along gradient '
     'directions; param2 is the center-vote threshold (lower = more circles, more false ones). minDist stops one coin '
     'being found twice; set it near the smallest coin diameter. The radius range rejects other round things.',
     ['circles'],
     {'MIN_DIST': (40, 'minimum distance between centers, pixels'), 'CANNY_HIGH': (120, 'param1: internal Canny high threshold'),
      'ACC_THRESHOLD': (30, 'param2: center votes needed'), 'MIN_RADIUS': (10, 'pixels'), 'MAX_RADIUS': (80, 'pixels')})

step('draw_circles', 'Draw circles and the count', ['img', 'circles'], ['vis'], """
vis = img.copy()
for x, y, r in circles:
    cv.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
cv.putText(vis, f'Count: {len(circles)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
stages['Result'] = vis
""", 'Outline + center for each detection, and the total written on the image.', ['drawing'])

# ---------------------------------------------------------------- thresholding / segmentation (Lab 05)
step('thresh_global', 'Global threshold', ['smooth'], ['mask'], """
_, mask = cv.threshold(smooth, THRESHOLD, 255, THRESH_TYPE)
stages['Global threshold'] = mask
""", 'One cutoff for the whole image: fine when lighting is even. THRESH_BINARY_INV makes dark objects white, '
     'which contour and morphology functions expect.', ['threshold'],
     {'THRESHOLD': (127, 'fixed cutoff 0-255'), 'THRESH_TYPE': ('cv.THRESH_BINARY', 'or cv.THRESH_BINARY_INV for dark objects')})

step('thresh_otsu', 'Otsu automatic threshold', ['smooth'], ['mask'], """
otsu_value, mask = cv.threshold(smooth, 0, 255, THRESH_TYPE | cv.THRESH_OTSU)
report['otsu_threshold'] = float(otsu_value)
stages[f'Otsu (T={otsu_value:.0f})'] = mask
""", 'Otsu picks the cutoff that best separates two intensity groups (minimum within-class variance), so it suits '
     'a two-peak (bimodal) histogram. The threshold argument 0 is ignored. Blurring first makes the peaks cleaner.',
     ['otsu'], {'THRESH_TYPE': ('cv.THRESH_BINARY', 'or cv.THRESH_BINARY_INV for dark objects')})

step('thresh_adaptive', 'Adaptive (local) threshold', ['smooth'], ['mask'], """
mask = cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_TYPE, BLOCK_SIZE, C)
stages['Adaptive threshold'] = mask
""", 'Each pixel gets its own cutoff from its neighbourhood (weighted mean minus C), so shadows and uneven light '
     'do not ruin it. BLOCK_SIZE must be odd and bigger than the strokes/objects; C shifts the cutoff.',
     ['adaptive'],
     {'BLOCK_SIZE': (31, 'odd neighbourhood size'), 'C': (10, 'constant subtracted from the local mean'),
      'THRESH_TYPE': ('cv.THRESH_BINARY', 'or cv.THRESH_BINARY_INV')})

step('hsv_range', 'Keep pixels inside the color range', ['hsv'], ['mask'], """
mask = cv.inRange(hsv, np.array(HSV_LOW), np.array(HSV_HIGH))
stages['Color mask'] = mask
""", 'inRange keeps pixels whose H, S and V all lie between the bounds. Saturation and value lower bounds stop '
     'gray and very dark pixels from matching any hue.', ['color_spaces', 'mask'],
     {'HSV_LOW': ([20, 100, 100], 'lower H,S,V'), 'HSV_HIGH': ([35, 255, 255], 'upper H,S,V')})

step('morph_clean', 'Clean the mask with opening and closing', ['mask'], ['mask'], """
kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)    # remove small specks
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)   # fill small holes
stages['Cleaned mask'] = mask
""", 'Opening (erode then dilate) removes specks smaller than the kernel; closing (dilate then erode) fills small '
     'holes and gaps. Kernel larger than the noise, smaller than the objects.', ['morphology'],
     {'MORPH_KSIZE': (5, 'structuring element size')})

step('contours', 'Find object outlines', ['mask'], ['contours'], """
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
report['count'] = len(contours)
report['areas'] = [int(cv.contourArea(c)) for c in contours]
""", 'findContours traces the boundary of each white blob; RETR_EXTERNAL keeps outer boundaries only. Small areas '
     'are noise, so they are dropped with MIN_AREA.', ['contours'],
     {'MIN_AREA': (100, 'smallest object area in pixels')})

step('draw_boxes', 'Draw a box and number around each object', ['img', 'contours'], ['vis'], """
vis = img.copy()
for i, c in enumerate(contours, 1):
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(vis, str(i), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
stages['Result'] = vis
""", 'Bounding boxes show where each object is and the number shows the count.', ['drawing', 'contours'])

step('watershed', 'Separate touching objects with marker-based watershed', ['img', 'mask'], ['markers'], """
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
""", 'Touching objects form one blob after thresholding. The distance transform peaks at each object center; '
     'thresholding it (DIST_FRACTION of the max) leaves one seed per object. Seeds are labelled, the unknown band '
     'is set to 0, and watershed floods from the seeds; boundaries come back as -1. Lower DIST_FRACTION if two '
     'objects share one seed, raise it if one object gets two. (The Lab 05 manual code uses 0.2 * max; that '
     'keeps touching objects joined, so 0.5 is used here to split them.)', ['watershed', 'distance', 'components'],
     {'DIST_FRACTION': (0.5, 'fraction of max distance for sure foreground')})

step('draw_watershed', 'Draw watershed boundaries', ['img', 'markers'], ['vis'], """
vis = img.copy()
vis[markers == -1] = (0, 0, 255)
cv.putText(vis, f"Objects: {report['count']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
stages['Result'] = vis
""", 'Pixels labelled -1 are the watershed lines between regions.', ['watershed'])

step('kmeans', 'Cluster pixel colors with k-means', ['img'], ['segmented'], """
pixels = img.reshape(-1, 3).astype(np.float32)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
segmented = {}
for k in K_VALUES:
    _, labels, centers = cv.kmeans(pixels, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
    segmented[k] = np.uint8(centers)[labels.flatten()].reshape(img.shape)
    stages[f'K = {k}'] = segmented[k]
    report[f'colors_k{k}'] = len(np.unique(labels))
""", 'k-means groups pixels into K color clusters and repaints each pixel with its cluster center. Data must be '
     'float32 of shape (N,3). Small K merges regions, large K splits them. Results depend on initialization, so '
     'attempts=3 with k-means++ is used.', ['kmeans'],
     {'K_VALUES': ([2, 4, 6], 'cluster counts to try')})

step('region_grow', 'Region growing from seed points', ['gray'], ['mask'], """
h, w = gray.shape
mask = np.zeros((h, w), np.uint8)
for tol in TOLERANCES:
    region = np.zeros((h, w), np.uint8)
    for sx, sy in SEEDS:
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv.floodFill(gray.copy(), ff, (sx, sy), 255, tol, tol,
                     4 | cv.FLOODFILL_MASK_ONLY | cv.FLOODFILL_FIXED_RANGE | (255 << 8))
        region |= ff[1:-1, 1:-1]
    stages[f'Region, tolerance {tol}'] = region
    report[f'area_tol{tol}'] = int((region > 0).sum())
    mask = region
""", 'Region growing starts at a seed and adds 4-connected neighbours whose intensity is within the tolerance of '
     'the seed (FIXED_RANGE compares to the seed, not to the neighbour). Small tolerance stops early; large '
     'tolerance leaks into the background.', ['region'],
     {'SEEDS': ([(50, 50)], 'seed points (x, y) inside the regions you want'),
      'TOLERANCES': ([5, 15, 30], 'intensity tolerances to compare')})

# ---------------------------------------------------------------- features and matching (Lab 04 / 06)
step('sift_both', 'Detect SIFT keypoints and descriptors in both images', ['ref', 'img'], ['kp_ref', 'des_ref', 'kp', 'des'], """
sift = cv.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(cv.cvtColor(ref, cv.COLOR_BGR2GRAY), None)
kp, des = sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)
report['keypoints_ref'] = len(kp_ref)
report['keypoints_scene'] = len(kp)
""", 'SIFT finds blob-like keypoints over a scale space (difference of Gaussians) and describes each with a '
     '128-value gradient histogram, so the same point matches across scale and rotation. Descriptors can be None '
     'for flat images.', ['sift'])

step('ratio_match', "Match descriptors with Lowe's ratio test", ['des_ref', 'des'], ['good'], """
good = []
if des_ref is not None and des is not None and len(des) >= 2:
    for pair in cv.BFMatcher(cv.NORM_L2).knnMatch(des_ref, des, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            good.append(pair[0])
report['good_matches'] = len(good)
""", 'For each reference descriptor take its two nearest scene descriptors (L2 distance for SIFT). Keep the match '
     'only if the best is clearly better than the second best (ratio < 0.75); ambiguous matches are dropped.',
     ['matching'], {'RATIO': (0.75, "Lowe's ratio; lower = stricter")})

step('homography', 'Estimate the homography with RANSAC', ['good', 'kp_ref', 'kp'], ['H', 'inliers'], """
H, inliers = None, 0
if len(good) >= 4:
    src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv.findHomography(src, dst, cv.RANSAC, RANSAC_PX)
    inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
report['inliers'] = inliers
report['found'] = H is not None and inliers >= MIN_INLIERS
""", 'A homography needs at least 4 point pairs. RANSAC fits it to random 4-pair samples and keeps the model most '
     'pairs agree with (error below RANSAC_PX), so wrong matches are ignored. Too few inliers = object not present.',
     ['ransac', 'perspective'],
     {'RANSAC_PX': (5.0, 'reprojection error allowed, pixels'), 'MIN_INLIERS': (10, 'inliers needed to accept')})

step('draw_object', 'Draw the recognised object outline', ['img', 'ref', 'H'], ['vis'], """
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
""", "The reference image's four corners are mapped through H into the scene; the polygon is the object's outline, "
     'which also shows its rotation and perspective.', ['drawing', 'sift'])

step('draw_matches', 'Show the good matches side by side', ['ref', 'kp_ref', 'img', 'kp', 'good'], [], """
stages['Matches'] = cv.drawMatches(ref, kp_ref, img, kp, good[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
""", 'Seeing the match lines is the fastest way to check whether the ratio test kept sensible pairs.', ['matching'])

step('stitch', 'Warp and blend the images into a panorama', ['imgs'], ['vis'], """
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
""", 'For each new image: SIFT keypoints, ratio-test matches, RANSAC homography that maps it onto the current '
     'panorama, then warp it onto a canvas big enough for both. The canvas is shifted so nothing gets cut off.',
     ['panorama', 'sift', 'ransac'], {'RATIO': (0.75, "Lowe's ratio")},
     helpers='''
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
''')

step('harris', 'Detect corners with Harris', ['gray'], ['corners'], """
response = cv.cornerHarris(np.float32(gray), 2, 3, HARRIS_K)
corners = np.argwhere(response > CORNER_FRACTION * response.max())
report['corners'] = len(corners)
""", 'Harris scores how much a small window changes when shifted in every direction; corners change in all '
     'directions. k (0.04-0.06) trades off edges vs corners.', ['corners'],
     {'HARRIS_K': (0.04, 'Harris k'), 'CORNER_FRACTION': (0.01, 'keep responses above this fraction of the max')})

step('gradients', 'Compute Sobel gradients, magnitude and direction', ['gray'], ['magnitude', 'direction'], """
gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gx ** 2 + gy ** 2)
direction = np.degrees(np.arctan2(gy, gx))
stages['Gradient magnitude'] = cv.convertScaleAbs(magnitude)
""", 'Sobel in x and y with a float depth (CV_64F) keeps negative slopes; magnitude = sqrt(gx^2+gy^2) and '
     'direction = atan2(gy, gx).', ['gradients', 'convert_abs'])

step('hog', 'HOG descriptor (shape / edge-direction features)', ['gray'], ['hog_vec'], """
patch = cv.resize(gray, (64, 128))
hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
hog_vec = hog.compute(patch).ravel()
report['hog_length'] = len(hog_vec)
""", 'HOG divides the window into 8x8 cells, builds a 9-bin gradient-direction histogram per cell and normalizes '
     'over 2x2-cell blocks; for 64x128 that is 7*15*36 = 3780 values. It describes shape and structure.', ['hog'])

step('lbp', 'LBP histogram (texture features)', ['gray'], ['lbp_hist'], """
lbp = lbp_image(gray)
lbp_hist = np.bincount(lbp.ravel(), minlength=256).astype(float)
lbp_hist /= lbp_hist.sum()
stages['LBP'] = lbp
report['lbp_bins'] = len(lbp_hist)
""", 'LBP compares each pixel with its 8 neighbours (1 if neighbour >= center) to make an 8-bit code; the '
     'histogram of codes describes texture and does not change with uniform brightness shifts.', ['lbp', 'texture'],
     helpers='''
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
''')

# ---------------------------------------------------------------- geometric transforms (Lab 03)
step('rotate', 'Rotate (and scale) about the center', ['img'], ['warped'], """
h, w = img.shape[:2]
M = cv.getRotationMatrix2D((w / 2, h / 2), ANGLE, SCALE)
warped = cv.warpAffine(img, M, (w, h))
report['matrix'] = np.round(M, 3).tolist()
stages[f'Rotated {ANGLE} deg'] = warped
""", 'getRotationMatrix2D builds the 2x3 matrix for rotation about a center (positive angle = counter-clockwise '
     'in the displayed image) with optional scale; warpAffine applies it.', ['affine'],
     {'ANGLE': (45, 'degrees, positive = counter-clockwise'), 'SCALE': (1.0, 'scale factor')})

step('translate', 'Translate (shift) the image', ['img'], ['warped'], """
h, w = img.shape[:2]
M = np.float32([[1, 0, TX], [0, 1, TY]])
warped = cv.warpAffine(img, M, (w, h))
stages['Translated'] = warped
""", 'Translation matrix [[1,0,tx],[0,1,ty]]: x moves by tx, y by ty (down is positive).', ['affine'],
     {'TX': (50, 'shift right, pixels'), 'TY': (30, 'shift down, pixels')})

step('shear', 'Shear (or undo a shear)', ['img'], ['warped'], """
h, w = img.shape[:2]
M = np.float32([[1, SHEAR_X, 0], [0, 1, 0]])
warped = cv.warpAffine(img, M, (int(w + abs(SHEAR_X) * h), h))
stages['Sheared'] = warped
""", 'Horizontal shear x\' = x + sh*y slants vertical lines; use the negative value to undo a known shear.',
     ['affine'], {'SHEAR_X': (-0.3, 'horizontal shear factor')})

step('affine3', 'Affine transform from three point pairs', ['img'], ['warped'], """
h, w = img.shape[:2]
M = cv.getAffineTransform(np.float32(SRC_POINTS[:3]), np.float32(DST_POINTS[:3]))
warped = cv.warpAffine(img, M, (w, h))
report['matrix'] = np.round(M, 3).tolist()
stages['Affine'] = warped
""", 'An affine map has 6 unknowns, so 3 non-collinear point pairs fix it. It keeps parallel lines parallel.',
     ['affine'], {'SRC_POINTS': ([[50, 50], [200, 50], [50, 200]], 'three source points'),
                  'DST_POINTS': ([[10, 100], [200, 50], [100, 250]], 'where they should go')})

step('perspective4', 'Perspective (top-down) rectification from four corners', ['img'], ['warped'], """
src = np.float32(SRC_POINTS)
dst = np.float32([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]])
P = cv.getPerspectiveTransform(src, dst)
warped = cv.warpPerspective(img, P, (OUT_W, OUT_H))
report['matrix'] = np.round(P, 4).tolist()
stages['Top-down view'] = warped
""", 'A homography has 8 unknowns, so 4 point pairs fix it. Give the corners in the same order as the output '
     'rectangle (top-left, top-right, bottom-right, bottom-left).', ['perspective'],
     {'SRC_POINTS': ([[60, 40], [340, 70], [360, 330], [40, 300]], 'four corners TL, TR, BR, BL'),
      'OUT_W': (400, 'output width'), 'OUT_H': (400, 'output height')})

# ---------------------------------------------------------------- change detection (video)
step('frame_diff', 'Compare each frame with the background', ['img', 'state'], ['mask'], """
g = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5, 5), 0)
if 'background' not in state:
    state['background'] = g          # first frame (scene empty) becomes the reference
diff = cv.absdiff(g, state['background'])
_, mask = cv.threshold(diff, DIFF_THRESHOLD, 255, cv.THRESH_BINARY)
mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
stages['Difference mask'] = mask
""", 'With a static camera, pixels that differ from an empty reference frame by more than DIFF_THRESHOLD are '
     'change. Blurring first stops sensor noise from counting; dilation joins broken blobs.', ['background'],
     {'DIFF_THRESHOLD': (30, 'gray-level change that counts as motion')})

step('zone_check', 'Raise an alert when change stays inside the restricted zone', ['mask', 'state'], ['contours'], """
zone = np.zeros_like(mask)
cv.fillPoly(zone, [np.int32(ZONE)], 255)
inside = cv.bitwise_and(mask, zone)
contours, _ = cv.findContours(inside, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv.contourArea(c) >= MIN_AREA]
state['streak'] = state.get('streak', 0) + 1 if contours else 0
report['alert'] = state['streak'] >= PERSIST_FRAMES
""", 'Only change inside the zone polygon matters. Requiring it for PERSIST_FRAMES frames in a row stops one noisy '
     'frame from raising an alarm.', ['background', 'mask'],
     {'ZONE': ([[100, 100], [300, 100], [300, 300], [100, 300]], 'restricted zone polygon (x, y)'),
      'MIN_AREA': (200, 'smallest changed area, pixels'), 'PERSIST_FRAMES': (3, 'frames in a row before alert')})

step('draw_zone', 'Draw zone, moving objects and alert', ['img', 'contours'], ['vis'], """
vis = img.copy()
cv.polylines(vis, [np.int32(ZONE)], True, (255, 0, 0), 2)
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
if report['alert']:
    cv.putText(vis, 'ALERT: zone entered', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
stages['Result'] = vis
""", 'Blue = zone, red boxes = changed regions inside it, text when the alert is active.', ['drawing'])

# ---------------------------------------------------------------- signals (wavelet anomaly task)
step('wavelet', 'Haar wavelet denoise and residual-based anomaly detection', ['signal'], ['denoised', 'anomalies'], """
x = np.asarray(signal, float)
approx, details = x[: len(x) // 2 ** LEVELS * 2 ** LEVELS], []
for _ in range(LEVELS):                                   # forward Haar transform
    details.append((approx[0::2] - approx[1::2]) / np.sqrt(2))
    approx = (approx[0::2] + approx[1::2]) / np.sqrt(2)
sigma = np.median(np.abs(details[0])) / 0.6745             # noise level from the finest details
thr = sigma * np.sqrt(2 * np.log(len(x)))                  # universal threshold
for d in reversed(details):                                # soft-threshold details, inverse transform
    d = np.sign(d) * np.maximum(np.abs(d) - thr, 0)
    up = np.empty(2 * len(approx)); up[0::2] = (approx + d) / np.sqrt(2); up[1::2] = (approx - d) / np.sqrt(2)
    approx = up
denoised = approx
residual = x[: len(denoised)] - denoised
mad = np.median(np.abs(residual - np.median(residual))) / 0.6745
anomalies = np.flatnonzero(np.abs(residual - np.median(residual)) > Z_LIMIT * max(mad, 1e-9))
report['anomalies'] = anomalies.tolist()
""", 'The wavelet transform splits the signal into a smooth part and detail parts. Small details are noise and are '
     'shrunk to zero; the smooth reconstruction is the normal behaviour. Points far from it (more than Z_LIMIT '
     'robust standard deviations) are anomalies. MAD is used instead of std so the spikes do not hide themselves.',
     ['wavelets', 'anomaly'],
     {'LEVELS': (3, 'wavelet decomposition levels'), 'Z_LIMIT': (4.0, 'robust z-score limit for an anomaly')})

# ---------------------------------------------------------------- basics (Lab 01)
step('blur_box', 'Blur the image with a large kernel', ['img'], ['blurred'], """
blurred = cv.blur(img, (BOX_KSIZE, BOX_KSIZE))
stages['Blurred'] = blurred
""", 'cv.blur averages each KxK neighbourhood (box filter); a 25x25 kernel gives a strong blur.', ['box'],
     {'BOX_KSIZE': (25, 'kernel size')})

step('center_roi', 'Crop the exact center region (ROI)', ['img'], ['roi'], """
h, w = img.shape[:2]
y0, x0 = (h - ROI_H) // 2, (w - ROI_W) // 2
roi = img[y0:y0 + ROI_H, x0:x0 + ROI_W]
stages['Center ROI'] = roi
""", 'NumPy slicing is [rows, cols] = [y, x]; the start is (size - roi) // 2 on each axis.', ['crop', 'coordinates'],
     {'ROI_W': (200, 'width'), 'ROI_H': (200, 'height')})

step('resize', 'Resize the image', ['img'], ['resized'], """
resized = cv.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR,
                    interpolation=cv.INTER_AREA if RESIZE_FACTOR < 1 else cv.INTER_CUBIC)
stages['Resized'] = resized
""", 'INTER_AREA is best for shrinking, INTER_CUBIC/LINEAR for enlarging.', ['resize'],
     {'RESIZE_FACTOR': (0.5, 'scale factor')})

step('draw_shapes', 'Draw shapes and text', ['img'], ['vis'], """
vis = img.copy()
h, w = vis.shape[:2]
cv.rectangle(vis, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 3)
cv.circle(vis, (w // 2, h // 2), min(h, w) // 6, (0, 0, 255), 3)
cv.putText(vis, TEXT, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
stages['Result'] = vis
""", 'Drawing functions take points as (x, y), colors as BGR, and thickness -1 to fill.', ['drawing'],
     {'TEXT': ('OpenCV', 'label text')})

step('blend', 'Blend a transparent banner', ['img'], ['vis'], """
vis = img.copy()
h, w = vis.shape[:2]
banner = vis.copy()
cv.rectangle(banner, (0, int(h * 0.85)), (w, h), (255, 0, 0), -1)
vis = cv.addWeighted(banner, ALPHA, vis, 1 - ALPHA, 0)
stages['Result'] = vis
""", 'addWeighted computes alpha*A + (1-alpha)*B; drawing on a copy and blending gives transparency.', ['blend'],
     {'ALPHA': (0.4, 'banner opacity')})

step('thresh_simple', 'Threshold the image', ['gray'], ['mask'], """
_, mask = cv.threshold(gray, THRESHOLD, 255, cv.THRESH_BINARY)
img = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)       # later steps work on the thresholded image
stages['Threshold'] = mask
""", 'cv.threshold returns (threshold used, image); pixels above THRESHOLD become 255, the rest 0.', ['threshold'],
     {'THRESHOLD': (127, 'cutoff 0-255')})

step('scale_matrix', 'Scale with a scaling matrix', ['img'], ['warped'], """
h, w = img.shape[:2]
M = np.float32([[SX, 0, 0], [0, SY, 0]])
warped = cv.warpAffine(img, M, (int(w * SX), int(h * SY)), flags=cv.INTER_CUBIC)
report['matrix'] = M.tolist()
stages['Scaled'] = warped
""", 'Scaling matrix [[sx,0,0],[0,sy,0]]: x is multiplied by sx and y by sy. The output size must be enlarged by '
     'the same factors or the result is cut off. Cubic interpolation fills the new pixels smoothly.', ['affine', 'resize'],
     {'SX': (2.0, 'horizontal scale'), 'SY': (2.0, 'vertical scale')})

step('fuse', 'Fuse the two registered images', ['ref', 'img'], ['vis'], """
if ref.shape != img.shape:
    img = cv.resize(img, (ref.shape[1], ref.shape[0]))   # registered slices must share one size
vis = cv.addWeighted(ref, WEIGHT, img, 1 - WEIGHT, 0)
stages['Fused'] = vis
stages['Result'] = vis
""", 'Weighted blending alpha*A + (1-alpha)*B keeps detail from both images. The images must already be '
     'registered (aligned pixel for pixel), otherwise the fusion shows double edges.', ['blend', 'registration'],
     {'WEIGHT': (0.5, 'weight of the first image (0-1)')})

step('channel_stats', 'Per-channel statistics', ['img'], [], """
for name, channel in zip('BGR', cv.split(img)):
    report[f'{name}_mean'] = round(float(channel.mean()), 2)
    report[f'{name}_std'] = round(float(channel.std()), 2)
    report[f'{name}_min_max'] = (int(channel.min()), int(channel.max()))
""", 'cv.split returns channels in B, G, R order.', ['channels', 'numpy'])

step('histogram_plot', 'Plot the intensity histogram', ['gray'], [], """
report['histogram'] = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()
""", 'calcHist with 256 bins over [0,256) counts pixels per gray level; it explains why a threshold method works '
     'or fails (two peaks suit Otsu).', ['histogram'])


# ---------------------------------------------------------------- step-specific warnings
# These replace the generic topic pitfalls for the step, so the advice matches the code shown.
WATCH = {
    'gaussian': 'Too much blur merges nearby edges (two screens, two lane dashes); too little lets noise vote.',
    'median': 'The kernel size must be odd and greater than 1 (3, 5, 7...).',
    'canny': 'Too low thresholds give texture edges that later vote for false lines/circles; too high break real boundaries.',
    'hough_p': 'HoughLinesP returns None when nothing passes the threshold, so check before indexing. Noisy edges '
               'break one side into several segments; raise MAX_LINE_GAP to join them.',
    'hv_filter': 'For a camera looking at the screens from an angle, borders are no longer horizontal/vertical; '
                 'raise ANGLE_TOLERANCE or rectify the view first.',
    'line_canvas': 'A closing kernel larger than the gap between neighbouring screens merges them into one shape.',
    'rects_from_canvas': 'A bounding rectangle is axis-aligned; for tilted screens use cv.minAreaRect instead.',
    'screen_state': 'Pick ON_LEVEL from your own image: print the brightness values and choose a value between '
                    'the dark and the lit group.',
    'missing_slots': 'The layout is learned from the screens that were found, so a whole missing row or column is '
                     'only caught when EXPECTED_SCREENS is set.',
    'lane_roi': 'If the camera position changes, the trapezoid must be moved too, or lane pixels get cut off.',
    'lane_fit': 'Fit x as a function of y; near-vertical lanes make y = m*x + c blow up.',
    'hough_circles': 'HoughCircles returns None when nothing is found and a (1, N, 3) float array otherwise. '
                     'Duplicate circles mean MIN_DIST is too small; missed coins mean ACC_THRESHOLD is too high.',
    'thresh_otsu': 'Otsu assumes two peaks in the histogram; with uneven lighting use adaptive thresholding.',
    'thresh_adaptive': 'BLOCK_SIZE must be odd and > 1; a block smaller than the strokes hollows out thick text.',
    'thresh_global': 'One cutoff fails under uneven lighting; compare with Otsu and adaptive.',
    'hsv_range': 'Red hue wraps around 0/180, so full red needs two ranges joined with cv.bitwise_or.',
    'morph_clean': 'A kernel bigger than the smallest object deletes it.',
    'contours': 'findContours expects white objects on black; invert the mask first if your objects are dark.',
    'watershed': 'cv.watershed needs a 3-channel image and int32 markers; it changes the markers array in place.',
    'kmeans': 'cv.kmeans needs float32 data shaped (N, 3) and returns labels shaped (N, 1); flatten before indexing.',
    'sift_both': 'cv.SIFT_create needs OpenCV 4.4 or newer (the patent expired); older builds put it in xfeatures2d.',
    'ratio_match': 'knnMatch can return fewer than 2 neighbours for some descriptors; check len(pair) == 2.',
    'homography': 'findHomography needs at least 4 matches and returns None when it fails.',
    'stitch': 'Images must overlap and be ordered; a wrong homography makes the warped image explode in size.',
    'frame_diff': 'Lighting changes and camera shake also count as change; keep the camera fixed and update the '
                  'background if light changes slowly.',
    'zone_check': 'The zone polygon is in pixel coordinates of this camera view; redraw it if the camera moves.',
    'clahe': 'CLAHE works on one channel; for color, apply it to the L channel of LAB.',
    'region_grow': 'floodFill needs a mask 2 pixels larger than the image in both directions.',
    'rotate': 'warpAffine keeps the original size, so rotated corners are cut off unless the output size is enlarged.',
    'perspective4': 'The four source points must be in the same order as the destination corners, or the image flips.',
    'wavelet': 'The signal length must be divisible by 2**LEVELS for this Haar code; extra samples at the end are dropped.',
}
for _id, _text in WATCH.items():
    S[_id]['watch'] = _text
