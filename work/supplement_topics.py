"""Supplemental related computer-vision topics beyond the supplied manuals.

Imported by build_content.py (topics + external references) and build_apis.py
(API symbol rows). Every code fragment runs under the standard topic context
(cv, np, image = BGR uint8, g = grayscale uint8) and is executed by
work/test_supplement.py before the knowledge base is rebuilt.
"""

PROVENANCE = ('Supplemental related topic beyond the supplied manuals; authored explanation, '
              'code executed by work/test_supplement.py against OpenCV 4.13. External refs are '
              'further reading, not quotation.')

# (id, title, aliases, answer, when_to_use, code, pitfalls, source_refs, related)
TOPICS = [
('fourier', 'Fourier transform and frequency-domain filtering',
 'FFT;DFT;np.fft.fft2;fftshift;ifft2;cv.dft;magnitude spectrum;low pass;high pass;frequency domain',
 'The 2D discrete Fourier transform represents an image as a sum of sinusoids. Low frequencies (near the centre after fftshift) carry smooth structure; high frequencies carry edges, texture and noise. Multiplying the spectrum by a mask and inverting gives frequency-domain filtering. By the convolution theorem, spatial convolution equals spectral multiplication.',
 'Inspect periodic noise, design ideal/Gaussian low- or high-pass filters, or explain why blurring removes detail.',
 "f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))\nmagnitude = 20 * np.log1p(np.abs(f))\nrows, cols = g.shape\ncy, cx = rows // 2, cols // 2\nmask = np.zeros((rows, cols), np.float32)\ncv.circle(mask, (cx, cy), 30, 1, -1)          # ideal low-pass\nlow = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))\nhigh = np.real(np.fft.ifft2(np.fft.ifftshift(f * (1 - mask))))\nlow_u8 = np.clip(low, 0, 255).astype(np.uint8)",
 'Display log magnitude, not raw magnitude. Forgetting ifftshift before ifft2 shifts the output. An ideal (hard-edged) mask causes ringing; a Gaussian mask avoids it. Take the real part and clip before converting to uint8.',
 'supplement:fourier', 'kernel;gaussian;wavelets'),

('pyramids', 'Image pyramids (Gaussian and Laplacian)',
 'pyrDown;pyrUp;Gaussian pyramid;Laplacian pyramid;multi-scale;image scale space',
 'A Gaussian pyramid repeatedly blurs and halves an image. A Laplacian pyramid stores the detail lost at each level (level minus upsampled next level), allowing exact reconstruction. Pyramids support coarse-to-fine search, multi-scale detection and seamless blending.',
 'Search for objects at several sizes, speed up processing on a coarse level, or blend two images without a visible seam.',
 "levels = [g]\nfor _ in range(3):\n    levels.append(cv.pyrDown(levels[-1]))\nlaplacian = []\nfor i in range(3):\n    h, w = levels[i].shape[:2]\n    up = cv.pyrUp(levels[i + 1], dstsize=(w, h))\n    laplacian.append(cv.subtract(levels[i], up, dtype=cv.CV_16S))\n# reconstruct the finest level from the next level plus stored detail\nh, w = levels[0].shape[:2]\nrebuilt = (cv.pyrUp(levels[1], dstsize=(w, h)).astype(np.int16) + laplacian[0]).astype(np.uint8)",
 'pyrUp does not undo pyrDown; detail is lost unless stored in the Laplacian level. Store Laplacian levels as signed types, since uint8 subtraction clips negatives. Pass dstsize for odd image sizes.',
 'supplement:pyramids', 'resize;gaussian;panorama'),

('template', 'Template matching',
 'matchTemplate;minMaxLoc;TM_CCOEFF_NORMED;TM_SQDIFF;find object;sliding window',
 'matchTemplate slides a template over the image and scores every position. With TM_CCOEFF_NORMED the best match is the maximum (1.0 is perfect); with TM_SQDIFF/TM_SQDIFF_NORMED the best match is the minimum. The result map is (H-h+1, W-w+1).',
 'Locate a known, same-scale, same-orientation pattern such as an icon, logo or fixed part.',
 "template = g[100:150, 20:100].copy()\nh, w = template.shape\nres = cv.matchTemplate(g, template, cv.TM_CCOEFF_NORMED)\n_, max_val, _, max_loc = cv.minMaxLoc(res)\ntop_left = max_loc\nbottom_right = (top_left[0] + w, top_left[1] + h)\nmarked = image.copy()\ncv.rectangle(marked, top_left, bottom_right, (0, 0, 255), 2)\n# all matches above a threshold\nys, xs = np.where(res >= 0.9)",
 'Not invariant to scale or rotation; use pyramids or features for those. For SQDIFF methods take min_loc, not max_loc. minMaxLoc returns (x, y) while np.where returns (rows, cols). Repeated detections near one location need non-maximum suppression.',
 'supplement:template', 'pyramids;sift;evaluation'),

('corners', 'Harris and Shi-Tomasi corner detection',
 'cornerHarris;goodFeaturesToTrack;Shi-Tomasi;cornerSubPix;corner detection;interest points',
 'Corners have strong intensity change in two directions. Harris computes a response R = det(M) - k*trace(M)^2 from the local gradient structure tensor M. Shi-Tomasi uses the smaller eigenvalue min(l1,l2) and goodFeaturesToTrack returns the strongest well-separated corners. cornerSubPix refines positions below pixel resolution.',
 'Choose points to track, calibrate with checkerboards, or explain what makes a good feature.',
 "gray32 = np.float32(g)\nR = cv.cornerHarris(gray32, blockSize=2, ksize=3, k=0.04)\nharris = image.copy()\nharris[R > 0.01 * R.max()] = (0, 0, 255)\npts = cv.goodFeaturesToTrack(g, maxCorners=50, qualityLevel=0.01, minDistance=10)\nif pts is not None:\n    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)\n    pts = cv.cornerSubPix(g, pts, (5, 5), (-1, -1), criteria)\n    for x, y in pts.reshape(-1, 2):\n        cv.circle(harris, (int(x), int(y)), 4, (0, 255, 0), 1)",
 'cornerHarris needs float32 input and returns a response map, not a point list; threshold relative to R.max(). Corners are not scale invariant. goodFeaturesToTrack may return None on flat images.',
 'supplement:corners', 'gradients;sift;optical_flow'),

('feature_detectors', 'FAST, AKAZE and BRISK feature detectors',
 'FastFeatureDetector_create;AKAZE_create;BRISK_create;FAST;AKAZE;BRISK;keypoint detector comparison',
 'FAST is a very quick corner detector without descriptors. AKAZE builds a nonlinear scale space and produces binary descriptors; BRISK produces binary descriptors with scale/rotation handling. Binary descriptors (ORB, AKAZE, BRISK) are matched with Hamming distance; float descriptors (SIFT) with L2.',
 'Choose a detector by speed, invariance, licensing and descriptor type.',
 "fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)\nkp_fast = fast.detect(g, None)\nakaze = cv.AKAZE_create()\nkp_a, des_a = akaze.detectAndCompute(g, None)\nbrisk = cv.BRISK_create()\nkp_b, des_b = brisk.detectAndCompute(g, None)\nvis = cv.drawKeypoints(image, kp_fast, None, color=(0, 255, 0))\nmatcher = cv.BFMatcher(cv.NORM_HAMMING)   # binary descriptors",
 'FAST alone gives no descriptor, so it cannot be matched without a separate extractor. Using NORM_L2 on binary descriptors gives meaningless distances. Descriptors can be None when no keypoints are found.',
 'supplement:features2d', 'orb;sift;matching;corners'),

('flann', 'FLANN matching and drawing matches',
 'FlannBasedMatcher;FLANN;drawMatches;drawMatchesKnn;approximate nearest neighbour;KD-tree;LSH',
 'FLANN performs approximate nearest-neighbour search, faster than brute force on large descriptor sets. For float descriptors (SIFT) use a KD-tree index; for binary descriptors use LSH. Combine knnMatch(k=2) with Lowe ratio test, then visualize with drawMatches.',
 'Match many descriptors quickly, for example in panoramas or object recognition.',
 "sift = cv.SIFT_create()\nimg2 = cv.warpAffine(g, cv.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), 10, 1.0), (g.shape[1], g.shape[0]))\nkp1, d1 = sift.detectAndCompute(g, None)\nkp2, d2 = sift.detectAndCompute(img2, None)\ngood = []\nif d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:\n    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))\n    for pair in flann.knnMatch(d1, d2, k=2):\n        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:\n            good.append(pair[0])\nvis = cv.drawMatches(g, kp1, img2, kp2, good[:30], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)",
 'FLANN KD-tree requires float32 descriptors; binary descriptors need algorithm=6 (LSH). knnMatch can return fewer than 2 neighbours, so check the pair length. Results are approximate and can vary slightly.',
 'supplement:matching', 'matching;sift;ransac'),

('shape', 'Moments, centroids and shape descriptors',
 'moments;centroid;HuMoments;convexHull;minAreaRect;boxPoints;fitEllipse;minEnclosingCircle;matchShapes;drawContours;aspect ratio;solidity;circularity',
 'Image moments summarise a contour: m00 is area, (m10/m00, m01/m00) is the centroid, and Hu moments are invariant to translation, scale and rotation. Shape measures such as aspect ratio, extent (area/bounding-box area), solidity (area/hull area) and circularity (4*pi*area/perimeter^2) help classify objects.',
 'Measure, filter or classify segmented objects, for example counting coins or rejecting non-rectangular screen candidates.',
 "_, bw = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)\ncontours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)\nout = image.copy()\nfor c in contours:\n    area = cv.contourArea(c)\n    if area < 50:\n        continue\n    M = cv.moments(c)\n    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])\n    hull = cv.convexHull(c)\n    solidity = area / max(cv.contourArea(hull), 1e-6)\n    circularity = 4 * np.pi * area / max(cv.arcLength(c, True) ** 2, 1e-6)\n    box = cv.boxPoints(cv.minAreaRect(c)).astype(np.int32)\n    cv.drawContours(out, [box], -1, (0, 255, 0), 2)\n    cv.circle(out, (cx, cy), 3, (0, 0, 255), -1)\n    hu = cv.HuMoments(M).ravel()",
 'Guard m00 == 0 before dividing. fitEllipse needs at least 5 points. findContours finds white objects on black, so invert if needed. Hu moments span many orders of magnitude; compare their log values.',
 'supplement:contours', 'contours;components;circles'),

('channels', 'Channels, flips, normalization, LUTs and borders',
 'split;merge;flip;rotate;transpose;normalize;NORM_MINMAX;LUT;lookup table;copyMakeBorder;BORDER_CONSTANT;BORDER_REPLICATE;BORDER_REFLECT_101;BORDER_WRAP;hconcat;vconcat',
 'split/merge separate and rebuild channels. flip mirrors (0 vertical, 1 horizontal, -1 both). normalize rescales values, e.g. NORM_MINMAX to 0..255. LUT maps every uint8 value through a 256-entry table, a fast way to apply gamma or custom curves. copyMakeBorder pads explicitly: CONSTANT (fixed value), REPLICATE (edge pixel), REFLECT (abc|cba), REFLECT_101 (abc|ba, the default), WRAP (periodic).',
 'Prepare inputs, visualize channel content, pad before filtering, or apply intensity curves efficiently.',
 "b, gch, r = cv.split(image)\nswapped = cv.merge([r, gch, b])\nmirror = cv.flip(image, 1)\nrot90 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)\nstretched = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)\ngamma = 0.5\ntable = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)\nbrighter = cv.LUT(g, table)\npadded = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0))\nside_by_side = cv.hconcat([g, brighter])",
 'cv.split is slower than NumPy indexing for a single channel. LUT tables must be uint8 of length 256. hconcat requires equal heights and types. Filters use BORDER_REFLECT_101 by default, not zero padding.',
 'supplement:border', 'padding;gamma;colors'),

('sharpening', 'Sharpening and unsharp masking',
 'sharpen;unsharp mask;sharpening kernel;high boost;detail enhancement',
 'Sharpening adds back high-frequency detail. Unsharp masking subtracts a blurred copy: sharp = image + amount*(image - blur), implemented with addWeighted. A 3x3 kernel [[0,-1,0],[-1,5,-1],[0,-1,0]] is the image plus a negative Laplacian.',
 'Enhance soft edges or text before display or detection.',
 "blur = cv.GaussianBlur(image, (0, 0), 3)\nunsharp = cv.addWeighted(image, 1.5, blur, -0.5, 0)\nkernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)\nsharp = cv.filter2D(image, -1, kernel)",
 'Sharpening amplifies noise; denoise first. Strong amounts create halos. Overshoot is clipped in uint8, so compute in float if exact values matter.',
 'supplement:filtering', 'kernel;laplacian;gaussian'),

('noise', 'Noise models, denoising and PSNR',
 'Gaussian noise;salt and pepper;speckle;fastNlMeansDenoising;fastNlMeansDenoisingColored;non-local means;PSNR;MSE;image quality',
 'Gaussian noise adds normally distributed values; salt-and-pepper sets random pixels to 0 or 255; speckle multiplies. Median filtering suits impulse noise, Gaussian/bilateral suit Gaussian noise, and non-local means averages similar patches anywhere nearby. PSNR = 10*log10(255^2/MSE) measures fidelity to a clean reference (higher is better).',
 'Create test inputs, choose a denoiser, and quantify the result.',
 "rng = np.random.default_rng(0)\nnoisy = np.clip(g.astype(np.float32) + rng.normal(0, 15, g.shape), 0, 255).astype(np.uint8)\nsp = g.copy()\nm = rng.random(g.shape)\nsp[m < 0.02] = 0\nsp[m > 0.98] = 255\nnlm = cv.fastNlMeansDenoising(noisy, None, h=15, templateWindowSize=7, searchWindowSize=21)\nmed = cv.medianBlur(sp, 3)\nprint('PSNR noisy', cv.PSNR(g, noisy), 'denoised', cv.PSNR(g, nlm))",
 'Add noise in float and clip, otherwise uint8 wraps around. PSNR needs a clean reference and does not always match perceived quality. Non-local means is slow on large images.',
 'supplement:denoising', 'median;bilateral;gaussian;evaluation'),

('grabcut', 'GrabCut foreground extraction',
 'grabCut;GC_INIT_WITH_RECT;GC_FGD;GC_BGD;GC_PR_FGD;foreground extraction;graph cut',
 'GrabCut models foreground and background colors with Gaussian mixture models and solves a graph cut. Initialize with a rectangle around the object (or a labelled mask), iterate, then keep pixels labelled definite or probable foreground.',
 'Interactive or box-guided object cut-out.',
 "mask = np.zeros(image.shape[:2], np.uint8)\nbgd = np.zeros((1, 65), np.float64)\nfgd = np.zeros((1, 65), np.float64)\nh, w = image.shape[:2]\nrect = (w // 8, h // 8, w * 3 // 4, h * 3 // 4)\ncv.grabCut(image, mask, rect, bgd, fgd, 3, cv.GC_INIT_WITH_RECT)\nfg = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)\ncutout = cv.bitwise_and(image, image, mask=fg)",
 'Input must be 8-bit 3-channel. The rectangle must be inside the image and contain the object. Similar foreground/background colors confuse the model; correct with mask hints and GC_INIT_WITH_MASK.',
 'supplement:grabcut', 'segmentation;watershed;mask'),

('inpaint', 'Image inpainting',
 'inpaint;INPAINT_TELEA;INPAINT_NS;restoration;remove scratches;remove object',
 'Inpainting fills masked pixels from their surroundings. INPAINT_TELEA uses fast marching; INPAINT_NS uses a Navier-Stokes based method. The mask is 8-bit with non-zero pixels marking damage.',
 'Remove scratches, text overlays or small unwanted objects.',
 "damaged = image.copy()\ndefect = np.zeros(image.shape[:2], np.uint8)\ncv.line(defect, (10, 10), (image.shape[1] - 10, image.shape[0] - 10), 255, 3)\ndamaged[defect > 0] = (255, 255, 255)\nrestored = cv.inpaint(damaged, defect, 3, cv.INPAINT_TELEA)",
 'Only plausible for thin or small regions; large holes become smeared. The mask must cover the whole defect, so dilate it slightly if needed.',
 'supplement:inpaint', 'noise;mask;morphology'),

('cascades', 'Haar cascade object detection',
 'CascadeClassifier;detectMultiScale;Haar cascade;Viola-Jones;face detection;cv.data.haarcascades',
 'A Haar cascade evaluates rectangular intensity-difference features in a sequence of increasingly strict stages, rejecting most windows early. detectMultiScale scans across positions and scales and returns (x, y, w, h) boxes. OpenCV ships trained frontal-face, eye and other cascades.',
 'Fast classical face or object detection on CPU when a trained cascade exists.',
 "path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'\nface = cv.CascadeClassifier(path)\nif face.empty():\n    raise FileNotFoundError(path)\nboxes = face.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))\nout = image.copy()\nfor (x, y, w, h) in boxes:\n    cv.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)",
 'Check classifier.empty(); a wrong path fails silently otherwise. Detection is sensitive to pose and lighting; equalizeHist can help. minNeighbors trades false positives for misses. Modern DNN detectors are usually more accurate.',
 'supplement:cascades', 'segmentation;dnn;evaluation'),

('background', 'Background subtraction and frame differencing',
 'createBackgroundSubtractorMOG2;createBackgroundSubtractorKNN;absdiff;frame difference;motion detection;foreground mask',
 'Frame differencing (absdiff between consecutive frames) detects change cheaply. MOG2 and KNN subtractors learn a per-pixel background model over time and return a foreground mask (MOG2 marks shadows as 127 when detectShadows=True). Morphology cleans the mask before contour analysis.',
 'Detect moving objects with a static camera, e.g. people entering a monitored zone.',
 "sub = cv.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)\nframes = [image] * 5 + [cv.rectangle(image.copy(), (20, 20), (80, 80), (255, 255, 255), -1)]\nfor frame in frames:\n    fgmask = sub.apply(frame)\n_, fgmask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)   # drop shadows (127)\nfgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))\ndiff = cv.absdiff(cv.cvtColor(frames[-2], cv.COLOR_BGR2GRAY), cv.cvtColor(frames[-1], cv.COLOR_BGR2GRAY))\n_, motion = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)",
 'Needs a static camera and warm-up frames. Lighting changes and moving backgrounds (trees, screens) cause false foreground. Threshold away shadow value 127 if shadows are not wanted.',
 'supplement:bgsub', 'video;morphology;contours'),

('optical_flow', 'Optical flow (Lucas-Kanade and Farneback)',
 'calcOpticalFlowPyrLK;calcOpticalFlowFarneback;Lucas-Kanade;sparse flow;dense flow;motion vectors;tracking',
 'Optical flow estimates apparent motion between frames assuming brightness constancy. Lucas-Kanade (sparse) tracks selected points, usually from goodFeaturesToTrack, using image pyramids. Farneback (dense) returns a 2-channel (dx, dy) vector for every pixel, often visualized as HSV with angle as hue and magnitude as value.',
 'Track points across frames, measure motion direction and speed.',
 "prev = g\nM = np.float32([[1, 0, 3], [0, 1, 2]])\nnxt = cv.warpAffine(g, M, (g.shape[1], g.shape[0]))\np0 = cv.goodFeaturesToTrack(prev, 50, 0.01, 7)\nif p0 is not None:\n    p1, st, err = cv.calcOpticalFlowPyrLK(prev, nxt, p0, None, winSize=(15, 15), maxLevel=2)\n    good_new, good_old = p1[st == 1], p0[st == 1]\nflow = cv.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)\nmag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])",
 'Points must be float32 of shape (N,1,2). Keep only status==1 results. Large motion, occlusion and lighting change break the brightness-constancy assumption; use pyramids (maxLevel) for larger motion.',
 'supplement:optflow', 'corners;video;background'),

('meanshift', 'Histogram back-projection, mean-shift and CamShift tracking',
 'calcBackProject;meanShift;CamShift;compareHist;HISTCMP_CORREL;HISTCMP_BHATTACHARYYA;color tracking;histogram comparison',
 'Back-projection replaces each pixel with the probability of its color under a target histogram (usually hue). meanShift moves a window to the local density peak of that map; CamShift also adapts window size and orientation. compareHist scores similarity between two histograms (correlation: higher is better; Bhattacharyya: lower is better).',
 'Track a colored object or compare color distributions of regions.',
 "hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)\nx, y, w, h = 40, 40, 60, 60\nroi = hsv[y:y + h, x:x + w]\nroi_hist = cv.calcHist([roi], [0], None, [180], [0, 180])\ncv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)\nback = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)\nterm = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)\n_, window = cv.meanShift(back, (x, y, w, h), term)\nrot_box, window = cv.CamShift(back, (x, y, w, h), term)\nh2 = cv.calcHist([hsv], [0], None, [180], [0, 180])\nsimilarity = cv.compareHist(roi_hist, h2, cv.HISTCMP_CORREL)",
 'Hue is unreliable for very dark or unsaturated pixels; mask them with inRange on S and V. Hue range in OpenCV is 0..179. Histograms passed to compareHist must have the same size and type.',
 'supplement:meanshift', 'histogram;colors;video'),

('color_spaces', 'LAB, YCrCb and HLS color spaces',
 'COLOR_BGR2LAB;COLOR_BGR2YCrCb;COLOR_BGR2HLS;CIELAB;luminance;chrominance;skin detection;color distance',
 'HSV separates hue from saturation/value. LAB approximates perceptual uniformity: L is lightness, a is green-red, b is blue-yellow, so Euclidean distance in LAB better reflects visible color difference. YCrCb separates luma (Y) from chroma (Cr, Cb) and is common for skin detection and for equalizing only brightness.',
 'Pick a color space where the property you threshold or compare is isolated.',
 "lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)\nL, A, B = cv.split(lab)\nclahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\nenhanced = cv.cvtColor(cv.merge([clahe.apply(L), A, B]), cv.COLOR_LAB2BGR)\nycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)\nskin = cv.inRange(ycrcb, (0, 133, 77), (255, 173, 127))\nref = np.array([[[0, 0, 255]]], np.uint8)\nref_lab = cv.cvtColor(ref, cv.COLOR_BGR2LAB).astype(np.float32)\ndist = np.linalg.norm(lab.astype(np.float32) - ref_lab, axis=2)",
 '8-bit LAB is scaled (L 0..255, a/b offset by 128), unlike textbook ranges. Skin thresholds are heuristic and vary across people and lighting. Convert back to BGR before display.',
 'supplement:colorspaces', 'colors;equalization;mask'),

('skeleton', 'Morphological skeletons and thinning',
 'skeleton;skeletonization;thinning;medial axis;hit-or-miss;MORPH_HITMISS',
 'A morphological skeleton reduces a binary shape to a one-pixel-wide centerline by repeatedly eroding and collecting the residue of an opening. It preserves topology approximately and is useful for measuring strokes, roads or cracks. The hit-or-miss transform finds exact local patterns.',
 'Measure thin structures, analyze handwriting or vessel/crack networks.',
 "_, bw = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)\nskel = np.zeros_like(bw)\nelement = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))\nwork = bw.copy()\nwhile cv.countNonZero(work) > 0:\n    opened = cv.morphologyEx(work, cv.MORPH_OPEN, element)\n    skel = cv.bitwise_or(skel, cv.subtract(work, opened))\n    work = cv.erode(work, element)",
 'Noise creates spurious branches; clean the mask first. This simple skeleton can be disconnected; dedicated thinning (opencv-contrib ximgproc.thinning) gives connected results but is not in the core package.',
 'supplement:morphology', 'morphology;morphology_more;distance'),

('stereo', 'Stereo disparity, epipolar geometry and depth',
 'StereoBM_create;StereoSGBM_create;disparity;depth map;findFundamentalMat;epipolar line;computeCorrespondEpilines;triangulation',
 'With two rectified cameras a point shifts horizontally between views; that shift is disparity d. Depth Z = f*B/d (focal length in pixels times baseline). StereoBM/StereoSGBM compute dense disparity. For unrectified views, the fundamental matrix F relates corresponding points via x2^T F x1 = 0 and defines epipolar lines.',
 'Estimate depth from two calibrated cameras or verify correspondences geometrically.',
 "shift = np.float32([[1, 0, -8], [0, 1, 0]])\nright = cv.warpAffine(g, shift, (g.shape[1], g.shape[0]))\nstereo = cv.StereoBM_create(numDisparities=32, blockSize=15)\ndisparity = stereo.compute(g, right).astype(np.float32) / 16.0\nf_px, baseline_m = 700.0, 0.12\nvalid = disparity > 0\ndepth = np.zeros_like(disparity)\ndepth[valid] = f_px * baseline_m / disparity[valid]",
 'StereoBM requires rectified 8-bit grayscale pairs; numDisparities must be divisible by 16 and blockSize odd. Output is fixed-point (divide by 16). Textureless areas give invalid disparity. Depth needs real calibration values.',
 'supplement:stereo', 'calibration;ransac;perspective'),

('dnn', 'Running pretrained neural networks with cv.dnn',
 'cv.dnn;readNet;readNetFromONNX;blobFromImage;setInput;forward;CNN inference;deep learning;YOLO;SSD',
 'OpenCV dnn runs pretrained models (ONNX, Caffe, TensorFlow, Darknet) on CPU without a training framework. blobFromImage resizes, scales, mean-subtracts, optionally swaps BGR to RGB and returns an NCHW float blob. The model weights are separate files that must be obtained and stored locally for offline use.',
 'Use a CNN classifier/detector/segmenter when handcrafted features are insufficient, still fully offline.',
 "blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(224, 224), mean=(0, 0, 0), swapRB=True, crop=False)\nassert blob.shape == (1, 3, 224, 224)\n# With a downloaded model file:\n# net = cv.dnn.readNetFromONNX('model.onnx')\n# net.setInput(blob)\n# scores = net.forward()",
 'Preprocessing (size, scale, mean, channel order) must match how the model was trained. No weights are bundled in this knowledge base. Large models may exceed the project size budget.',
 'supplement:dnn', 'advanced;cascades;segmentation'),

('plotting', 'Matplotlib figures for image analysis',
 'plt.figure;figsize;plt.title;plt.colorbar;cmap;plt.axis off;plt.imsave;plt.plot intensity profile;plt.xlabel;plt.legend;suptitle;plt.show',
 'Matplotlib figures compare processing stages. Use subplots for grids, set cmap="gray" with vmin/vmax for single-channel images, add colorbar for signed or float maps (e.g. gradients, distance transform), and plot 1D intensity profiles or histograms with labelled axes and legends. savefig writes a file; show opens a window.',
 'Produce clear, labelled visual evidence for lab reports and debugging.',
 "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nsx = cv.Sobel(g, cv.CV_64F, 1, 0, ksize=3)\nfig, axes = plt.subplots(1, 3, figsize=(12, 4))\naxes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)); axes[0].set_title('Original')\nim = axes[1].imshow(sx, cmap='seismic'); axes[1].set_title('Sobel x (signed)')\nfig.colorbar(im, ax=axes[1])\naxes[2].plot(g[g.shape[0] // 2], label='middle row'); axes[2].set_xlabel('x'); axes[2].legend()\nfor a in axes[:2]: a.axis('off')\nfig.suptitle('Processing stages'); fig.tight_layout()\nfig.savefig('stages.png', dpi=100); plt.close(fig)",
 'Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable. Close figures in loops to free memory. plt.show blocks in scripts and does nothing with the Agg backend.',
 'supplement:matplotlib', 'display;histogram;numpy'),
]

EXTERNAL = [
 ('fourier', 'OpenCV Fourier transform tutorial', 'https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html', 'FFT, spectrum display and frequency filtering.'),
 ('pyramids', 'OpenCV image pyramids', 'https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html', 'Gaussian/Laplacian pyramids and blending.'),
 ('template', 'OpenCV template matching', 'https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html', 'Scoring methods and multiple matches.'),
 ('corners', 'OpenCV Harris corners', 'https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html', 'Harris response and sub-pixel refinement.'),
 ('features2d', 'OpenCV features2d module', 'https://docs.opencv.org/4.x/da/d9b/group__features2d.html', 'Detector/descriptor classes such as FAST, AKAZE, BRISK.'),
 ('denoising', 'OpenCV image denoising', 'https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html', 'Non-local means parameters.'),
 ('grabcut', 'OpenCV GrabCut', 'https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html', 'Rectangle and mask initialization.'),
 ('inpaint', 'OpenCV inpainting', 'https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html', 'TELEA and Navier-Stokes methods.'),
 ('cascades', 'OpenCV cascade classifier', 'https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html', 'Haar cascade detection usage.'),
 ('bgsub', 'OpenCV background subtraction', 'https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html', 'MOG2/KNN subtractors.'),
 ('optflow', 'OpenCV optical flow', 'https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html', 'Lucas-Kanade and Farneback flow.'),
 ('meanshift', 'OpenCV meanshift and camshift', 'https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html', 'Back-projection based tracking.'),
 ('colorspaces', 'OpenCV color conversions', 'https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html', 'LAB, YCrCb and HLS formulas and ranges.'),
 ('stereo', 'OpenCV depth map from stereo', 'https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html', 'StereoBM disparity computation.'),
 ('dnn', 'OpenCV dnn module', 'https://docs.opencv.org/4.x/d6/d0f/group__dnn.html', 'Model loading and blobFromImage preprocessing.'),
 ('matplotlib', 'Matplotlib pyplot reference', 'https://matplotlib.org/stable/api/pyplot_summary.html', 'Figure, axes, colorbar and saving functions.'),
]

# name|topic|inputs|outputs  (added to apis.jsonl; symbol existence verified at build time)
API_ROWS = '''dft|fourier|float32 single-channel image; DFT_COMPLEX_OUTPUT flag|Two-channel complex spectrum
pyrDown|pyramids|Image and optional dstsize|Blurred half-size image
pyrUp|pyramids|Image and optional dstsize|Upsampled double-size image
matchTemplate|template|Image, smaller template, method|Score map of size (H-h+1, W-w+1)
minMaxLoc|template|Single-channel array, optional mask|minVal, maxVal, minLoc(x,y), maxLoc(x,y)
cornerHarris|corners|float32 gray, blockSize, ksize, k|Harris response map
goodFeaturesToTrack|corners|Gray, maxCorners, qualityLevel, minDistance|float32 points (N,1,2) or None
cornerSubPix|corners|Gray, initial corners, window, zeroZone, criteria|Refined corners
FastFeatureDetector_create|feature_detectors|threshold, nonmaxSuppression|FAST detector (detect only)
AKAZE_create|feature_detectors|Optional parameters|AKAZE detector/descriptor
BRISK_create|feature_detectors|Optional parameters|BRISK detector/descriptor
FlannBasedMatcher|flann|indexParams dict, searchParams dict|Approximate matcher
drawMatches|flann|img1, kp1, img2, kp2, matches, outImg, flags|Side-by-side match visualization
moments|shape|Contour or binary image|Dict of spatial/central/normalized moments
HuMoments|shape|Moments dict|Seven invariant moments
convexHull|shape|Point set|Hull points or indices
minAreaRect|shape|Point set|((cx,cy),(w,h),angle)
boxPoints|shape|Rotated rect|Four float corner points
fitEllipse|shape|At least five points|Rotated rect describing ellipse
minEnclosingCircle|shape|Point set|((cx,cy),radius)
matchShapes|shape|Two contours, method, parameter|Dissimilarity (0 = identical)
drawContours|shape|Image, contour list, index (-1 all), color, thickness|Drawn image (in place)
split|channels|Multi-channel image|Tuple of single-channel arrays
merge|channels|List of same-size single-channel arrays|Multi-channel image
flip|channels|Image, flipCode 0/1/-1|Mirrored image
rotate|channels|Image, ROTATE_90_CLOCKWISE etc.|Rotated image
normalize|channels|Array, dst, alpha, beta, norm type|Rescaled array
LUT|channels|uint8 image, 256-entry uint8 table|Mapped image
copyMakeBorder|channels|Image, top, bottom, left, right, borderType, value|Padded image
hconcat|channels|List of equal-height images|Horizontally joined image
fastNlMeansDenoising|noise|Gray image, h, templateWindowSize, searchWindowSize|Denoised image
fastNlMeansDenoisingColored|noise|BGR image, h, hColor, windows|Denoised color image
PSNR|noise|Reference and test arrays of equal size|Peak signal-to-noise ratio in dB
grabCut|grabcut|BGR image, mask, rect, bgdModel, fgdModel, iterations, mode|Updates mask in place
inpaint|inpaint|Image, 8-bit mask, radius, method|Restored image
CascadeClassifier|cascades|Path to trained XML|Classifier with detectMultiScale/empty
createBackgroundSubtractorMOG2|background|history, varThreshold, detectShadows|Subtractor with apply()
createBackgroundSubtractorKNN|background|history, dist2Threshold, detectShadows|Subtractor with apply()
absdiff|background|Two equal-size arrays|Absolute difference
calcOpticalFlowPyrLK|optical_flow|prev, next, prevPts float32 (N,1,2), nextPts, winSize, maxLevel|nextPts, status, err
calcOpticalFlowFarneback|optical_flow|prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags|Dense (H,W,2) flow
cartToPolar|optical_flow|x and y components|Magnitude and angle
calcBackProject|meanshift|Images, channels, histogram, ranges, scale|Probability map
meanShift|meanshift|Probability map, window, criteria|Iterations and new window
CamShift|meanshift|Probability map, window, criteria|Rotated box and new window
compareHist|meanshift|Two histograms, method|Similarity or distance score
StereoBM_create|stereo|numDisparities (multiple of 16), blockSize (odd)|Block-matching stereo object
StereoSGBM_create|stereo|minDisparity, numDisparities, blockSize, ...|Semi-global stereo object
findFundamentalMat|stereo|Two point arrays, method|Fundamental matrix and inlier mask
'''
