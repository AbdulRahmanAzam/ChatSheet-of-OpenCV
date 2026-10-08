"""Prerequisite code that makes each topic's code fragment self-contained.

Topic fragments are written as they would appear mid-program (e.g. `mask` already
exists). SETUPS supplies the minimal earlier steps; build_content.py stores it on
each topic as `setup` plus `requires` (the names it defines), and
work/test_topics.py executes setup + code for every topic.

Standard context for every topic: cv, np, image (BGR uint8), g (grayscale uint8).
Fragments that import from `cv_core`/`lab06` need outputs/cv_knowledge_base/solutions
on sys.path.
"""

_MASK = "_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)"
_POINTS = ("src_points = np.float32([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40], [20, 60], [80, 10], [30, 25]])\n"
           "dst_points = src_points * 1.2 + np.float32([15, 8])")
_SIGNAL = ("rng = np.random.default_rng(0)\n"
           "clean_baseline = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)\n"
           "signal = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)\n"
           "signal[100] += 3.0   # injected impulse")

# topic id -> (setup code, names it defines)
SETUPS = {
    'pixels': ("x, y = 10, 20", ['x', 'y']),
    'io': ("cv.imwrite('input.png', image)   # stand-in for your own image file", ['input.png']),
    'numpy': ("dx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)\ndy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)", ['dx', 'dy']),
    'crop': ("image = cv.resize(image, (640, 480))   # crop needs an image of at least 300x300", ['image']),
    'resize': (_MASK, ['mask']),
    'mask': ("hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)", ['hsv']),
    'padding': ("k = np.ones((3, 3), np.float32) / 9", ['k']),
    'convert_abs': ("dx = cv.Sobel(g, cv.CV_16S, 1, 0, ksize=3)", ['dx']),
    'morphology': (_MASK, ['mask']),
    'distance': (_MASK, ['mask']),
    'components': (_MASK, ['mask']),
    'contours': (_MASK, ['mask']),
    'coordinates': ("x, y = 10.0, 20.0\nH = np.array([[1.0, 0.1, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])", ['x', 'y', 'H']),
    'registration': (_POINTS, ['src_points', 'dst_points']),
    'ransac': (_POINTS, ['src_points', 'dst_points']),
    'matching': ("sift = cv.SIFT_create()\n"
                 "rotated = cv.warpAffine(g, cv.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), 8, 1.0), (g.shape[1], g.shape[0]))\n"
                 "_, desc1 = sift.detectAndCompute(g, None)\n"
                 "_, desc2 = sift.detectAndCompute(rotated, None)", ['desc1', 'desc2']),
    'panorama': ("rng = np.random.default_rng(1)\n"
                 "scene = cv.GaussianBlur(rng.integers(0, 256, (240, 480, 3), dtype=np.uint8), (5, 5), 0)\n"
                 "image1, image2 = scene[:, :300].copy(), scene[:, 180:].copy()   # overlapping views", ['image1', 'image2']),
    'wavelets': (_SIGNAL, ['signal']),
    'anomaly': (_SIGNAL, ['signal', 'clean_baseline']),
    'video': ("h, w = image.shape[:2]\n"
              "writer = cv.VideoWriter('input.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (w, h))\n"
              "for i in range(5):\n    writer.write(np.roll(image, 4 * i, axis=1))\n"
              "writer.release()\n"
              "def frame_function(frame):\n    return cv.Canny(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 50, 150)",
              ['input.mp4', 'frame_function']),
    'evaluation': (_MASK + "\npredicted = mask\ntruth = np.roll(mask, 3, axis=1)", ['predicted', 'truth']),
    'calibration': ("h, w = image.shape[:2]\n"
                    "camera_matrix = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])\n"
                    "distortion_coefficients = np.array([-0.1, 0.01, 0, 0, 0])   # from cv.calibrateCamera in practice",
                    ['camera_matrix', 'distortion_coefficients']),
}

# Fragments that import the bundled reference solutions.
NEEDS_SOLUTIONS = {'padding', 'wavelets', 'anomaly', 'video', 'panorama'}

# Plain-English paraphrases users type, added to topic aliases for retrieval.
EXTRA_ALIASES = {
    'io': ['read image', 'load image from disk', 'open image file', 'save image'],
    'display': ['show image', 'colors look wrong', 'blue tint', 'image looks blue'],
    'resize': ['half size', 'scale down', 'shrink', 'make smaller', 'make bigger', 'enlarge', 'upscale', 'downscale'],
    'mask': ['select color', 'yellow pixels', 'color range', 'color detection', 'extract color', 'only red pixels'],
    'bilateral': ['keep edges sharp', 'smooth but keep edges', 'denoise preserve edges'],
    'adaptive': ['uneven lighting', 'shadows', 'varying illumination', 'non uniform lighting', 'document scan threshold'],
    'gaussian': ['blur', 'smooth', 'soften'],
    'canny': ['find edges', 'edge map'],
    'hough': ['detect straight lines', 'line detection', 'lane lines'],
    'circles': ['detect circles', 'round objects', 'balls'],
    'components': ['count objects', 'label blobs', 'blob counting'],
    'contours': ['outline', 'object boundary', 'shape outline'],
    'perspective': ['bird eye view', 'top down view', 'deskew document', 'scan document'],
    'equalization': ['dark image', 'low contrast', 'improve contrast'],
    'drawing': ['write text', 'draw box', 'annotate image', 'bounding box'],
}
