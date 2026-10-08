"""Recipes: which steps solve which kind of task, and the words that point to it.

A recipe is an ordered list of step ids (see plan_steps.py). Slot entries such as
'<threshold>' are filled from the task text (e.g. "using Otsu" -> thresh_otsu).
Optional entries ('?id') are added only when the task asks for them.

cues: (regex, weight). One regex counts once however often it matches.
  3 = the requested operation or the method itself (rotate, stitch, Hough, SIFT, watershed)
  2 = a typical object or goal of that method (coins, lanes, monitors, sensor readings)
  1 = context that only hints (road, document, medical)
detect=True: the recipe finds things, so without a finding verb (detect, find, count, ...) its
score is halved; "draw five circles" is a drawing task, not circle detection.
"""

R = {}


def recipe(id, title, kind, steps, cues, goal, video_ok=False, topics=(), limits='', params=None, detect=False):
    R[id] = dict(id=id, title=title, kind=kind, steps=steps, cues=cues, goal=goal, video_ok=video_ok,
                 topics=list(topics), limits=limits, params=params or {}, detect=detect)


# Colour words, but not when they describe the background or something the task draws.
COLOR_WORD = r'\b(yellow|red|green|blue|orange|purple|pink|white|black)\b(?!\s+(background|banner|box|rectangle|text|line|circle|border))'

recipe('screens', 'Screen detection and status with Hough lines', 'image',
       ['gray', 'gaussian', 'canny', 'hough_p', 'hv_filter', 'line_canvas', 'rects_from_canvas',
        'screen_state', 'missing_slots', 'draw_screens'],
       [(r'\bscreens?\b|\bmonitors\b|\b(a|the|each|every|computer) monitor\b', 3),
        (r'switched (on|off)|\bon or off\b|\bon/off\b|which .*\bare (switched )?on\b|powered', 2),
        (r'computer lab|rows of (computers|pcs)|\bpcs?\b|exam hall|office', 1), (r'\bmissing\b', 1)],
       'find every screen, say whether it is ON or OFF, and report missing screens', video_ok=True,
       topics=['screen', 'hough'],
       limits='Assumes a roughly front-facing camera so screen borders are near horizontal/vertical. Brightness '
              'shows whether a screen is lit, not whether the computer is powered (a dark desktop looks OFF, glare '
              'looks ON). A missing screen is inferred from a gap in the row/column layout, so a person blocking a '
              'screen also looks missing.',
       params={'BLUR_KSIZE': 7, 'HOUGH_THRESHOLD': 40, 'MIN_LINE_LENGTH': 40, 'MAX_LINE_GAP': 25})

recipe('lanes', 'Lane-line detection with Hough lines', 'image',
       ['gray', 'gaussian', 'canny', 'lane_roi', 'hough_p', 'lane_fit', 'draw_lanes'],
       [(r'\blanes?\b', 3), (r'lane markings?|lane boundar', 2),
        (r'\broad\b|highway|dashcam|autonomous|self-driving|driver[- ]assist', 1)],
       'find the left and right lane lines and draw them', video_ok=True, topics=['hough'],
       params={'HOUGH_THRESHOLD': 20, 'MIN_LINE_LENGTH': 20, 'MAX_LINE_GAP': 60}, detect=True)   # dashed markings

recipe('lines', 'Straight-line detection with Hough', 'image',
       ['gray', 'gaussian', 'canny', 'hough_p', 'draw_lines'],
       [(r'hough line|houghlines|line transform', 3), (r'straight', 2),
        (r'\blines?\b|\brails?\b|railway|tracks?\b|\bshelves\b|facade|court', 1)],
       'detect straight line segments and draw them', video_ok=True, topics=['hough'], detect=True)

recipe('circles', 'Circle detection and counting with Hough circles', 'image',
       ['gray', 'median', 'hough_circles', 'draw_circles'],
       [(r'hough circles?|houghcircles|circle transform', 3),
        (r'\bcoins?\b|\bcircles?\b|circular|\bround\b|\bballs?\b|\bwheels?\b|\biris\b|\bpupils?\b|\bcaps?\b|\bholes?\b|manholes?', 2)],
       'detect each circle, draw it and count them', video_ok=True, topics=['circles'], detect=True)

recipe('recognize', 'Object recognition with SIFT, ratio test and RANSAC homography', 'pair',
       ['sift_both', 'ratio_match', 'homography', 'draw_object', '?draw_matches'],
       [(r'\bsift\b|\borb\b|keypoints?|feature[- ]match|matching', 3),
        (r'reference (image|photo|object)|recogni[sz]|identify|logo|template|known (object|logo)|given a photo of|photos? of each', 2),
        (r'different (scales?|angles)|orientations?|occlusion|partly hidden|partially hidden|rotated|crowded', 2),
        (r'\bassets?\b|inventory|which of', 1)],
       'find the reference object in the scene and outline it', video_ok=True,
       topics=['sift', 'matching', 'ransac'], detect=True)

recipe('panorama', 'Panorama stitching with SIFT and homography', 'images',
       ['stitch'],
       [(r'panoram|stitch', 3), (r'(merge|join|combine)\b.*\b(photos|images|pictures|views|halves)', 3),
        (r'overlap\w* (photos|images|pictures|views|shots)|with (an )?overlap|panning', 2),
        (r'side by side|wide (image|picture)|single wide', 2)],
       'align overlapping images and stitch them into one panorama', topics=['panorama'])

recipe('count', 'Object counting with threshold and contours', 'image',
       ['gray', 'gaussian', '<threshold>', 'morph_clean', 'contours', 'draw_boxes'],
       [(r'contours?|blobs?|connected components?', 3), (r'\bcount\w*|how many|number of', 2),
        (r'\bareas?\b|measure|outline each', 2),
        (r'\bcells?\b|\bpills?\b|\bobjects\b|\bnuts\b|\bbolts\b|\bwashers?\b|\bparticles\b|\bleaf|\bleaves\b|\bparts\b', 1)],
       'separate the objects from the background, outline and count them', video_ok=True,
       topics=['threshold', 'contours', 'morphology'], detect=True)

recipe('watershed', 'Marker-based watershed segmentation', 'image',
       ['gray', 'gaussian', 'thresh_otsu', 'watershed', 'draw_watershed'],
       [(r'watershed', 3), (r'touch|clump|stuck together', 3),
        (r'overlap\w* (each other|objects|coins|cells|grains|nuclei)|(cells|coins|grains|nuclei) (that )?overlap', 3),
        (r'\bseparat', 2), (r'distance[- ]transform|markers?', 2)],
       'split touching objects into separate regions and count them', topics=['watershed'], detect=True)

recipe('kmeans', 'Color clustering with k-means', 'image',
       ['kmeans'],
       [(r'k-?means|clustering|\bclusters\b', 3),
        (r'\bk\s*=|quantiz|dominant colou?rs|\d+ colou?rs|group (the )?pixels|similar colou?r|land-cover', 2)],
       'group pixels into K color clusters and compare K values', topics=['kmeans'])

recipe('color', 'Color segmentation in HSV', 'image',
       ['hsv', 'hsv_range', 'morph_clean', 'contours', 'draw_boxes'],
       [(r'\bhsv\b|colou?r (segmentation|range|threshold)|inrange', 3), (COLOR_WORD, 2), (r'isolat|extract', 1)],
       'keep only pixels of the target color and outline the objects', video_ok=True,
       topics=['color_spaces', 'mask'], detect=True)

recipe('threshold', 'Thresholding: global vs Otsu vs adaptive', 'image',
       ['gray', 'gaussian', 'thresh_global', 'thresh_otsu', 'thresh_adaptive', '?histogram_plot'],
       [(r'adaptive|otsu|threshold|binari[sz]|black[- ]and[- ]white', 3), (r'uneven|shadow|lamp|bimodal', 2),
        (r'document|\bpage\b|\btext\b|handwrit|notes', 1)],
       'turn the image into foreground/background and compare threshold methods', topics=['threshold', 'adaptive', 'otsu'])

recipe('edges', 'Edge detection with Canny', 'image',
       ['gray', 'gaussian', 'canny', '?gradients'],
       [(r'\bcanny\b|edge detect', 3), (r'\bedges?\b|boundar(y|ies)', 2), (r'sobel|gradient|outline', 1)],
       'find the object edges and compare thresholds', video_ok=True, topics=['canny', 'gradients'])

recipe('region', 'Region growing from seeds', 'image',
       ['gray', 'region_grow'],
       [(r'region[- ]grow|\bgrow', 3), (r'\bseeds?\b|flood ?fill|clicked|starting from a (seed )?(pixel|point)', 2)],
       'grow regions from seed points and compare tolerances', topics=['region'])

recipe('zone', 'Restricted-zone change monitoring', 'video',
       ['frame_diff', 'morph_clean', 'zone_check', 'draw_zone'],
       [(r'restricted|intru|unauthori[sz]ed|danger area|no-parking|forbidden', 3),
        (r'motion|\bmov(es|ing|ement)\b|walks? into|\benters?\b|entering|background subtraction|frame differenc|change detection', 3),
        (r'\bzone\b|cctv|security camera|surveillance|alarm|alert', 2)],
       'detect anything that enters the zone and raise an alert', topics=['background'])

recipe('enhance', 'Contrast enhancement and pseudocolor', 'image',
       ['gray', 'clahe', '?equalize', '?gamma', '?log', '?pseudocolor', 'enhance_report'],
       [(r'enhanc|contrast|clahe|equali[sz]|gamma|brighten|visib|washed out|underexposed', 3),
        (r'pseudo-?colou?r|colou?r ?map|false colou?r', 2),
        (r'x-?ray|\bmri\b|\bct\b|medical|ultrasound|echo|thermal|night|\bdark\b|dull', 1)],
       'make the important detail visible and measure the improvement', video_ok=True,
       topics=['equalization', 'gamma'])

recipe('fusion', 'Image fusion of two registered images', 'pair',
       ['fuse'],
       [(r'\bfus(e|ion|ing)\b', 3), (r'registered|\bct\b.*\bmri\b|\bmri\b.*\bct\b', 2)],
       'blend two aligned images so both kinds of detail are visible', topics=['blend'])

recipe('transform', 'Geometric transformation', 'image',
       ['<transform>'],
       [(r'rotat|tilt|crooked|upright|orientation', 3), (r'translat|\bshift|\bmove the image', 3), (r'shear|slant|skew', 3),
        (r'affine|three (landmark )?points', 3),
        (r'perspective|top-?down|from (directly )?above|bird|rectif|four (corners|points)|homograph|flat\b|straighten|from the side', 3),
        (r'scal(e|ing) matrix|\bmatrix\b|enlarg|\bwarp', 2), (r'transform', 1)],
       'apply or undo the geometric transformation', topics=['affine', 'perspective'])

recipe('texture', 'Texture and shape features (HOG and LBP)', 'image',
       ['gray', 'gradients', 'hog', 'lbp'],
       [(r'\bhog\b|\blbp\b|local binary|histogram of oriented', 3), (r'texture|material|fabric|rough|surfaces?', 2),
        (r'descriptors?|features?', 1)],
       'describe the texture/shape with feature vectors', topics=['hog', 'lbp', 'texture'])

recipe('corners', 'Corner detection with Harris', 'image',
       ['gray', 'harris'],
       [(r'harris|corner (points|detect)|detect (the )?corners|find (the )?(\w+ )?corners', 3), (r'corners?', 1)],
       'find corner points', topics=['corners'])

recipe('wavelet', 'Wavelet denoising and anomaly detection on a signal', 'signal',
       ['wavelet'],
       [(r'wavelet|\bhaar\b', 3), (r'sensor|signal|time series|readings|\blog\b|heartbeat|\becg\b|pressure|temperature', 2),
        (r'anomal|abnormal|spikes?|irregular|unusual|peaks?|malfunction', 2)],
       'denoise the signal and flag abnormal points', topics=['wavelets', 'anomaly'])

# Basic operations are scored per operation asked for (see planner.understand), not by cues.
recipe('basics', 'Basic image operations', 'image',
       ['?gray', '?blur_box', '?center_roi', '?resize', '?thresh_simple', '?rotate', '?draw_shapes', '?blend',
        '?channel_stats', '?histogram_plot'],
       [(r'\bload\b|\bdisplay\b|\bshow\b|title|axes|\bread\b', 1)],
       'carry out the requested basic operations', topics=['io', 'display'])

BASIC_OPS = {
    'blur_box': r'\bblur', 'center_roi': r'\broi\b|crop|region of interest|central \d+|center region',
    'resize': r'resiz|shrink', 'thresh_simple': r'threshold', 'rotate': r'rotat', 'draw_shapes': r'\bdraw\b',
    'blend': r'banner|transparen|blend', 'channel_stats': r'statistic|average|\bmean\b|standard deviation|\bstd\b',
}

# Verbs that make a detection recipe applicable.
DETECT_VERBS = (r'detect|\bfind|count|how many|locat|identif|recogni|segment|isolat|extract|measure|check|track|'
                r'separat|spot|report|monitor|\bmark|distinguish|classif|\bwhere\b|outline|split|highlight')

# Slot choices: (regex, step). The first match wins; the last entry is the default.
SLOTS = {
    '<threshold>': [(r'adaptive|uneven|shadow', 'thresh_adaptive'), (r'global|fixed threshold|threshold (of|=)\s*\d', 'thresh_global'),
                    (r'', 'thresh_otsu')],
    '<transform>': [(r'perspective|top-?down|from (directly )?above|bird|rectif|four (corners|points)|homograph|flat\b|from the side', 'perspective4'),
                    (r'affine|three (landmark )?points', 'affine3'), (r'shear|slant|skew', 'shear'),
                    (r'translat|\bshift|\bmove the image', 'translate'), (r'rotat|tilt|crooked|upright|orientation|degrees', 'rotate'),
                    (r'scal|enlarg|zoom|shrink', 'scale_matrix'), (r'', 'rotate')],
}

# Optional steps: added when their cue appears in the task.
OPTIONAL = dict(BASIC_OPS, **{
    'draw_matches': r'match|correspond', 'histogram_plot': r'histogram', 'gradients': r'gradient|sobel',
    'equalize': r'equali[sz]|histogram', 'gamma': r'gamma|dark|brighten|underexposed', 'log': r'\blog\b',
    'pseudocolor': r'pseudo|colou?r ?map|false colou?r|heat ?map|visuali[sz]', 'gray': r'gray|grey|threshold|histogram',
})

VIDEO_CUES = r'\bvideos?\b|webcam|camera feed|live (feed|camera)|real-?time|\bframes?\b|stream|footage|cctv'

# HSV ranges for named colors (OpenCV hue 0-179). Red wraps around 0, so it uses the low band here.
COLORS = {
    'red': ([0, 120, 70], [10, 255, 255]), 'orange': ([10, 120, 80], [20, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]), 'green': ([36, 80, 50], [85, 255, 255]),
    'blue': ([90, 80, 50], [130, 255, 255]), 'purple': ([130, 60, 50], [160, 255, 255]),
    'pink': ([160, 60, 80], [179, 255, 255]), 'white': ([0, 0, 200], [179, 40, 255]),
    'black': ([0, 0, 0], [179, 255, 50]),
}

# Worked lab tasks (KB tasks.jsonl) that each recipe corresponds to; the closest one is attached to the plan.
RECIPE_TASKS = {
    'screens': ['L06-T01'], 'lanes': ['L06-T06'], 'circles': ['L06-T07'], 'recognize': ['L06-T04', 'L06-T02'],
    'panorama': ['L06-T05', 'L03-T09'], 'zone': ['L06-T08'], 'wavelet': ['L06-T03'],
    'watershed': ['L05-T07', 'L05-T08'], 'kmeans': ['L05-T09'], 'color': ['L05-T04'],
    'threshold': ['L05-T01', 'L05-T02', 'L05-T03', 'L05-T10'], 'edges': ['L05-T05'], 'region': ['L05-T06'],
    'enhance': ['L02-T01', 'L02-T03'], 'fusion': ['L02-T02'], 'texture': ['L04-T01'],
    'transform': ['L03-T01', 'L03-T02', 'L03-T03', 'L03-T04', 'L03-T05', 'L03-T06', 'L03-T07', 'L03-T08', 'L03-T10',
                  'L01-T07'],
    'basics': ['L01-T03', 'L01-T04', 'L01-T05', 'L01-T06', 'L01-T07', 'L01-T08', 'L01-T09'],
}

# KB topic -> recipe, used when no cue fires and the KB search has to decide.
TOPIC_TO_RECIPE = {
    'hough': 'lines', 'screen': 'screens', 'circles': 'circles', 'sift': 'recognize', 'matching': 'recognize',
    'ransac': 'recognize', 'orb': 'recognize', 'flann': 'recognize', 'template': 'recognize', 'panorama': 'panorama',
    'registration': 'panorama', 'contours': 'count', 'components': 'count', 'morphology': 'count',
    'watershed': 'watershed', 'distance': 'watershed', 'kmeans': 'kmeans', 'segmentation': 'threshold',
    'color_spaces': 'color', 'threshold': 'threshold', 'adaptive': 'threshold', 'otsu': 'threshold',
    'canny': 'edges', 'gradients': 'edges', 'scharr': 'edges', 'laplacian': 'edges', 'region': 'region',
    'background': 'zone', 'video': 'zone', 'equalization': 'enhance', 'gamma': 'enhance', 'pseudocolor': 'enhance',
    'histogram': 'enhance', 'affine': 'transform', 'perspective': 'transform', 'hog': 'texture', 'lbp': 'texture',
    'texture': 'texture', 'gabor': 'texture', 'corners': 'corners', 'wavelets': 'wavelet', 'anomaly': 'wavelet',
    'blend': 'fusion', 'io': 'basics', 'display': 'basics', 'crop': 'basics', 'drawing': 'basics', 'channels': 'basics',
}
