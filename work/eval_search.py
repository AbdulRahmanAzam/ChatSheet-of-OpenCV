"""Retrieval check: realistic user questions must return the right topic in the top 3.

Queries are hand-written (not copied from qa.jsonl) and include the user's own
spellings. Writes outputs/cv_knowledge_base/reports/search_eval.json.
"""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'cv_knowledge_base'
sys.path.insert(0, str(ROOT))
from search import search  # noqa: E402

CASES = [
    ('how do I read an image from disk', 'io'),
    ('show image with matplotlib colors look wrong blue', 'display'),
    ('convert VGA to RGB then grayscale', 'colors'),
    ('what does ddepth -1 mean in filter2D', 'depth'),
    ('crop the center of the picture R10', 'crop'),
    ('make picture half size', 'resize'),
    ('write text on image', 'drawing'),
    ('blend two images with transparency', 'blend'),
    ('select only yellow pixels', 'mask'),
    ('plot dot bar of pixel counts', 'histogram'),
    ('improve contrast of dark photo', 'equalization'),
    ('gamma correction', 'gamma'),
    ('difference between convolution and correlation', 'kernel'),
    ('border types reflect padding', 'padding'),
    ('botch filter', 'box'),
    ('getGaussianKernel 2D', 'gaussian'),
    ('remove salt and pepper noise', 'median'),
    ('smooth but keep edges sharp', 'bilateral'),
    ('sobel x and y gradient magnitude', 'gradients'),
    ('scar edge detection', 'scharr'),
    ('laplacian of gaussian zero crossing', 'laplacian'),
    ('convert scale absolute', 'convert_abs'),
    ('canny thresholds hysteresis', 'canny'),
    ('binary threshold inverse', 'threshold'),
    ('uneven lighting document threshold', 'adaptive'),
    ('automatic threshold value', 'otsu'),
    ('dilate and erode noise in mask', 'morphology'),
    ('top hat black hat', 'morphology_more'),
    ('region growing from a seed point', 'region'),
    ('distance transform sure foreground', 'distance'),
    ('count objects connected components', 'components'),
    ('separate touching coins watershed markers', 'watershed'),
    ('k means color clustering', 'kmeans'),
    ('find contours and approximate polygon', 'contours'),
    ('homogeneous coordinates 3x3 matrix', 'coordinates'),
    ('rotate and shear affine', 'affine'),
    ('bird eye view four corners warp', 'perspective'),
    ('align CT and MRI images', 'registration'),
    ('histogram of oriented gradients HOG', 'hog'),
    ('local binary pattern texture', 'lbp'),
    ('edge direction histogram atan2', 'hed'),
    ('classify wood metal fabric texture', 'texture'),
    ('gabor filter', 'gabor'),
    ('hough line transformation detect lines', 'hough'),
    ('monitor computer lab screens missing on off', 'screen'),
    ('detect coins circles', 'circles'),
    ('sift create detect and compute draw keypoints', 'sift'),
    ('bf matcher ratio test', 'matching'),
    ('outliers in matches findHomography', 'ransac'),
    ('ORB features', 'orb'),
    ('stitch two photos panorama', 'panorama'),
    ('haar wavelet denoising', 'wavelets'),
    ('process every frame of a video', 'video'),
    ('intersection over union accuracy', 'evaluation'),
    ('remove lens distortion', 'calibration'),
    ('fft low pass filter', 'fourier'),
    ('image pyramid pyrDown', 'pyramids'),
    ('find a small logo inside a big image', 'template'),
    ('harris corner detection', 'corners'),
    ('akaze brisk fast detector', 'feature_detectors'),
    ('flann matcher', 'flann'),
    ('centroid of a contour moments', 'shape'),
    ('split channels and merge', 'channels'),
    ('sharpen blurry image', 'sharpening'),
    ('add gaussian noise and measure psnr', 'noise'),
    ('cut out foreground object grabcut', 'grabcut'),
    ('remove scratches from old photo', 'inpaint'),
    ('face detection haar cascade', 'cascades'),
    ('detect moving people static camera', 'background'),
    ('lucas kanade tracking points', 'optical_flow'),
    ('camshift color tracking', 'meanshift'),
    ('LAB color space perceptual', 'color_spaces'),
    ('skeleton of a shape', 'skeleton'),
    ('depth from two cameras disparity', 'stereo'),
    ('run onnx model blobFromImage', 'dnn'),
    ('colorbar and subplot figure', 'plotting'),
]


# Written after tuning CASES and never used for tuning aliases/search; measures generalization.
HELDOUT = [('load a jpg file','io'),('picture too big reduce dimensions','resize'),('keep only green color area','mask'),
('noise removal without blurring edges','bilateral'),('threshold page photographed with shadow','adaptive'),
('blur the photo','gaussian'),('outline of edges in photo','canny'),('find road lane lines','hough'),
('how many objects are in the binary image','components'),('flatten a tilted document','perspective'),
('brighten underexposed photo contrast','equalization'),('put a rectangle around the face','drawing'),
('find the circle shaped coins','circles'),('match keypoints between two images','matching'),
('segment overlapping cells','watershed'),('compare texture of surfaces','texture'),
('track an object across video frames','optical_flow'),('smooth with median filter','median'),
('second derivative edges','laplacian'),('image upside down mirror','channels')]


def run(cases):
    rows = []
    for query, expected in cases:
        got = [r['id'].split(':', 1)[1] for r in search(query, limit=60) if r['kind'] == 'topic'][:3]
        rows.append(dict(query=query, expected=expected, top3=got, rank=got.index(expected) + 1 if expected in got else None))
    return dict(cases=len(rows), top1=sum(r['rank'] == 1 for r in rows), top3=sum(r['rank'] is not None for r in rows), results=rows)


def main():
    tuned, held = run(CASES), run(HELDOUT)
    report = dict(tuned=tuned, heldout=held,
                  note=('tuned: queries used while adding aliases/stemming (optimistic). heldout: unseen paraphrases '
                        '(realistic). Lexical BM25 misses paraphrases with no shared words; the planner stage should '
                        'add a learned task classifier.'))
    (ROOT / 'reports' / 'search_eval.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, r in (('tuned', tuned), ('heldout', held)):
        print(f"{name}: top1 {r['top1']}/{r['cases']}  top3 {r['top3']}/{r['cases']}")


if __name__ == '__main__':
    main()
