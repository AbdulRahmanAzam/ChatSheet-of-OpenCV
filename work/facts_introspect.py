"""Facts and triples read directly from the installed OpenCV/NumPy (exact by construction).

Generates constant values, algorithm defaults and return structures by calling the
library, so nothing is hand-typed. Imported by build_content.py.
"""
import cv2 as cv
import numpy as np

SOURCE = 'opencv_introspection'

CONSTANT_GROUPS = {
    'imread flag': ['IMREAD_UNCHANGED', 'IMREAD_GRAYSCALE', 'IMREAD_COLOR', 'IMREAD_ANYDEPTH', 'IMREAD_ANYCOLOR', 'IMREAD_REDUCED_GRAYSCALE_2'],
    'color conversion code': ['COLOR_BGR2GRAY', 'COLOR_BGR2RGB', 'COLOR_RGB2BGR', 'COLOR_BGR2HSV', 'COLOR_HSV2BGR', 'COLOR_BGR2LAB',
                              'COLOR_GRAY2BGR', 'COLOR_RGB2GRAY', 'COLOR_BGR2YCrCb', 'COLOR_BGR2HLS', 'COLOR_BGR2XYZ', 'COLOR_BGRA2BGR'],
    'threshold type': ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV', 'THRESH_OTSU', 'THRESH_TRIANGLE'],
    'adaptive threshold method': ['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C'],
    'morphology operation': ['MORPH_ERODE', 'MORPH_DILATE', 'MORPH_OPEN', 'MORPH_CLOSE', 'MORPH_GRADIENT', 'MORPH_TOPHAT', 'MORPH_BLACKHAT'],
    'structuring element shape': ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE'],
    'distance type': ['DIST_L1', 'DIST_L2', 'DIST_C'],
    'image depth': ['CV_8U', 'CV_8S', 'CV_16U', 'CV_16S', 'CV_32S', 'CV_32F', 'CV_64F'],
    'Hough method': ['HOUGH_STANDARD', 'HOUGH_PROBABILISTIC', 'HOUGH_MULTI_SCALE', 'HOUGH_GRADIENT', 'HOUGH_GRADIENT_ALT'],
    'border type': ['BORDER_CONSTANT', 'BORDER_REPLICATE', 'BORDER_REFLECT', 'BORDER_WRAP', 'BORDER_REFLECT_101', 'BORDER_DEFAULT'],
    'interpolation flag': ['INTER_NEAREST', 'INTER_LINEAR', 'INTER_CUBIC', 'INTER_AREA', 'INTER_LANCZOS4'],
    'kmeans flag': ['KMEANS_RANDOM_CENTERS', 'KMEANS_PP_CENTERS', 'KMEANS_USE_INITIAL_LABELS'],
    'termination criteria type': ['TERM_CRITERIA_COUNT', 'TERM_CRITERIA_MAX_ITER', 'TERM_CRITERIA_EPS'],
    'norm type': ['NORM_L1', 'NORM_L2', 'NORM_HAMMING', 'NORM_INF'],
    'drawKeypoints flag': ['DRAW_MATCHES_FLAGS_DEFAULT', 'DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG',
                           'DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS', 'DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS'],
    'rotate code': ['ROTATE_90_CLOCKWISE', 'ROTATE_180', 'ROTATE_90_COUNTERCLOCKWISE'],
}
DEPTH_NAMES = {getattr(cv, n): n for n in ['CV_8U', 'CV_8S', 'CV_16U', 'CV_16S', 'CV_32S', 'CV_32F', 'CV_64F']}


def build():
    """Return (facts, triples). facts: (ref, topic, statement, note); triples: (subjects, attribute, value, ref)."""
    facts, triples = [], []
    ref = f'{SOURCE}:opencv-{cv.__version__}'
    for group, names in CONSTANT_GROUPS.items():
        for name in names:
            if not hasattr(cv, name):
                continue
            value = int(getattr(cv, name))
            facts.append((ref, 'api', f'The integer value of cv2.{name} ({group}) is {value}.', ''))
            triples.append(([f'cv2.{name}', name], 'integer value', str(value), ref))

    sift = cv.SIFT_create()
    defaults = {'nfeatures': sift.getNFeatures(), 'nOctaveLayers': sift.getNOctaveLayers(),
                'contrastThreshold': round(sift.getContrastThreshold(), 4), 'edgeThreshold': round(sift.getEdgeThreshold(), 4),
                'sigma': round(sift.getSigma(), 4)}
    for k, v in defaults.items():
        v = int(v) if float(v).is_integer() and k != 'sigma' else v
        facts.append((ref, 'sift', f'The default {k} of cv2.SIFT_create is {v}.', ''))
        triples.append((['SIFT_create', 'SIFT'], f'default {k}', str(v), ref))
    dtype = DEPTH_NAMES.get(sift.descriptorType(), str(sift.descriptorType()))
    facts.append((ref, 'sift', f'The default descriptorType of cv2.SIFT_create is {dtype} (float descriptors), descriptor size {sift.descriptorSize()}.', ''))
    triples.append((['SIFT_create', 'SIFT'], 'default descriptorType', dtype, ref))
    triples.append((['SIFT descriptor', 'SIFT'], 'descriptor size', str(sift.descriptorSize()), ref))

    orb = cv.ORB_create()
    facts.append((ref, 'orb', f'The default nfeatures of cv2.ORB_create is {orb.getMaxFeatures()}; ORB descriptors are {orb.descriptorSize()} bytes of type '
                  f'{DEPTH_NAMES.get(orb.descriptorType())} (binary, matched with NORM_HAMMING).', ''))
    triples.append((['ORB_create', 'ORB'], 'default nfeatures', str(orb.getMaxFeatures()), ref))

    # Return structures, observed by calling the functions.
    g = (np.arange(64 * 64) % 256).astype(np.uint8).reshape(64, 64)
    r = cv.threshold(g, 100, 255, cv.THRESH_BINARY)
    facts.append((ref, 'threshold', f'cv2.threshold returns {len(r)} values: retval (the threshold used, a float) and dst (the thresholded image).', ''))
    triples.append((['cv2.threshold', 'threshold()'], 'returns', 'retval and dst', ref))
    data = np.float32(np.random.default_rng(0).random((50, 3)))
    r = cv.kmeans(data, 3, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv.KMEANS_RANDOM_CENTERS)
    facts.append((ref, 'kmeans', f'cv2.kmeans returns {len(r)} values: compactness, labels (one per sample, shape (N,1)) and centers (shape (K, features)).', ''))
    triples.append((['cv2.kmeans', 'kmeans()'], 'returns', 'compactness, labels, centers', ref))
    r = cv.connectedComponents((g > 128).astype(np.uint8))
    facts.append((ref, 'components', f'cv2.connectedComponents returns {len(r)} values: retval (number of labels including background 0) and the labels image (markers).', ''))
    triples.append((['cv2.connectedComponents', 'connectedComponents'], 'returns', 'retval (count) and labels/markers image', ref))
    kp, des = cv.SIFT_create().detectAndCompute(cv.GaussianBlur(np.random.default_rng(0).integers(0, 255, (128, 128), dtype=np.uint8), (5, 5), 0), None)
    if des is not None:
        facts.append((ref, 'sift', f'sift.detectAndCompute returns keypoints and descriptors; descriptors has shape (N, {des.shape[1]}) of dtype {des.dtype} for N keypoints.', ''))
        triples.append((['SIFT descriptors', 'descriptors'], 'shape', f'(N, {des.shape[1]})', ref))
    return facts, triples
