"""Conceptual "why / how" facts for the course topics that the manuals state only briefly or not at all.

Standard computer-vision / OpenCV knowledge, marked 'cv_fundamentals:<topic>' so answers show it is not quoted
from the course manual. Format as elsewhere: (ref, topic, statement, note).
"""
F = 'cv_fundamentals:'

FACTS = [
# ---- smoothing before edges / lines / circles
(F + 'canny', 'canny', 'Images are blurred before Canny because derivatives amplify noise: each noisy pixel creates a strong local gradient that would become a false edge, so smoothing first leaves only real intensity changes.', ''),
(F + 'canny', 'canny', 'Canny uses two thresholds (hysteresis) so that weak edge pixels are kept only when connected to strong ones; this keeps edges continuous while rejecting isolated noise responses.', ''),
(F + 'canny', 'canny', 'Non-maximum suppression in Canny keeps a pixel only if its gradient magnitude is the largest along the gradient direction, which thins wide gradient ridges into one-pixel-wide edges.', ''),
(F + 'circles', 'circles', 'HoughCircles is blurred first (often with a median blur) because noise and texture create many edge pixels that vote for false centers; median blur removes speckle noise while keeping the circle boundaries sharp.', ''),
(F + 'hough', 'hough', 'The Hough line transform represents a line as rho = x*cos(theta) + y*sin(theta); every edge pixel votes for all (rho, theta) lines through it, and accumulator cells with many votes are the detected lines.', ''),
(F + 'hough', 'hough', 'The (rho, theta) form is used instead of y = mx + c because the slope m is infinite for vertical lines, while rho and theta are bounded for every line.', ''),
(F + 'hough', 'hough', 'HoughLines returns infinite lines as (rho, theta); HoughLinesP (probabilistic) returns finite segments (x1, y1, x2, y2) and is faster because it uses a random subset of edge points.', ''),
(F + 'hough', 'hough', 'Hough is applied to an edge map (e.g. Canny output), not to the raw image, because only boundary pixels should vote.', ''),
(F + 'hough', 'hough', 'cv2.HoughLinesP returns a line segment only if it satisfies both conditions: its line accumulates at least threshold votes, and the segment is at least minLineLength pixels long.', ''),
(F + 'hough', 'hough', 'Hough line detection finds lines of any orientation (theta from 0 to pi): horizontal, vertical and diagonal lines can all be detected.', ''),
(F + 'hough', 'hough', 'minLineLength in HoughLinesP is a lower limit: segments shorter than it are discarded; any segment at least that long can be returned.', ''),
(F + 'hough', 'hough', 'HoughLinesP outputs line segments as endpoint coordinates (x1, y1, x2, y2).', ''),
(F + 'hough', 'hough', 'In HoughLinesP the threshold is the minimum number of accumulator votes (edge pixels on the line); a higher threshold keeps only stronger, usually longer lines.', ''),
(F + 'circles', 'circles', 'cv2.HoughCircles returns an array of circles, each as (x, y, radius) of its center and radius, or None when no circle is found.', ''),
(F + 'canny', 'canny', 'cv2.Canny returns a single-channel grayscale binary edge map: 255 on edge pixels and 0 elsewhere.', ''),
(F + 'color_spaces', 'color_spaces', 'cv2.inRange returns a single-channel binary mask (255 where the pixel is inside the range, 0 elsewhere); to see the original colours of the kept pixels use cv2.bitwise_and(image, image, mask=mask).', ''),
(F + 'circles', 'circles', 'cv2.HoughCircles returns a circle only if its center gets at least param2 accumulator votes and its radius lies between minRadius and maxRadius; centers closer than minDist to a stronger circle are dropped.', ''),
# ---- thresholding
(F + 'otsu', 'otsu', 'Otsu tries every threshold and picks the one that minimizes the weighted within-class variance (equivalently maximizes the between-class variance) of the two pixel groups it creates.', ''),
(F + 'otsu', 'otsu', 'Otsu works best when the histogram is bimodal (two clear peaks: object and background); with uneven lighting or one dominant peak it fails, and adaptive thresholding is used instead.', ''),
(F + 'adaptive', 'adaptive', 'Adaptive thresholding computes a separate threshold for every pixel from its neighbourhood (mean or Gaussian-weighted mean minus C), so shadows and uneven illumination do not break the segmentation as they do for one global threshold.', ''),
# ---- morphology / contours
(F + 'morphology', 'morphology', 'Opening (erosion then dilation) removes small white noise specks; closing (dilation then erosion) fills small black holes and gaps inside objects.', ''),
(F + 'contours', 'contours', 'findContours expects a binary image with white objects on a black background; RETR_EXTERNAL returns only the outer boundaries and CHAIN_APPROX_SIMPLE stores only the corner points of straight runs.', ''),
# ---- watershed
(F + 'watershed', 'watershed', 'The distance transform gives each foreground pixel its distance to the nearest background pixel, so the centers of objects become peaks; thresholding the peaks gives one separate seed (sure foreground) per object even when objects touch.', ''),
(F + 'watershed', 'watershed', 'Marker-based watershed is used because plain watershed over-segments: every small local minimum from noise floods into its own region; markers restrict flooding to one region per marked object.', ''),
(F + 'watershed', 'watershed', 'Watershed treats the image as a topographic surface and floods it from the markers; where floods from different markers meet, a boundary (label -1 in OpenCV) is built.', ''),
(F + 'watershed', 'watershed', 'Before watershed the unknown region (sure background minus sure foreground) is set to 0 in the markers, telling watershed that these pixels are still to be decided; background is labelled 1 so it is not confused with unknown.', ''),
# ---- k-means
(F + 'kmeans', 'kmeans', 'cv2.kmeans requires float32 data because it computes means and distances in floating point; uint8 pixel values must be reshaped to (N, 3) and converted with np.float32 first.', ''),
(F + 'kmeans', 'kmeans', 'k-means repeats two steps until the centers stop moving: assign each pixel to the nearest center, then move each center to the mean of its assigned pixels.', ''),
(F + 'kmeans', 'kmeans', 'k-means needs K chosen in advance and depends on the initial centers, so several attempts (or k-means++ initialisation) are used and the most compact result is kept.', ''),
# ---- region growing
(F + 'region', 'region', 'Region growing starts from seed pixels and repeatedly adds neighbouring pixels whose intensity is similar (within a tolerance) to the region or seed, stopping when no neighbour qualifies.', ''),
# ---- colour
(F + 'color_spaces', 'color_spaces', 'HSV is used for colour segmentation because hue describes the colour independently of brightness, so one hue range finds an object in both shadow and light, which is hard with BGR values.', ''),
(F + 'colors', 'colors', 'Images are converted to grayscale before edge detection because edges are changes in intensity: one intensity channel is enough, it is three times less data to process, and gradient and threshold functions such as Canny expect a single-channel image.', ''),
(F + 'colors', 'colors', 'Images are converted from BGR to RGB before plt.imshow because OpenCV stores channels as BGR while Matplotlib expects RGB; without conversion red and blue are swapped.', ''),
# ---- SIFT and matching
(F + 'matching', 'matching', "Lowe's ratio test keeps a match only if the distance to the best match is clearly smaller than to the second-best (ratio below about 0.75); ambiguous matches, where two candidates are almost equally similar, are rejected.", ''),
(F + 'matching', 'matching', 'SIFT descriptors are matched with the L2 (Euclidean) norm; binary descriptors such as ORB use the Hamming norm.', ''),
(F + 'matching', 'matching', 'crossCheck=True in BFMatcher keeps a match only if the two descriptors are each other\'s best match in both directions.', ''),
(F + 'sift', 'sift', 'SIFT is scale invariant because keypoints are found as extrema across a Gaussian scale space (Difference of Gaussians), and rotation invariant because each descriptor is computed relative to the keypoint\'s dominant gradient orientation.', ''),
(F + 'sift', 'sift', 'A SIFT descriptor is 128 values: a 4x4 grid of cells around the keypoint, each with an 8-bin gradient-orientation histogram.', ''),
(F + 'sift', 'sift', 'SIFT discards low-contrast keypoints (contrast threshold) and keypoints lying on edges (edge threshold, using the ratio of principal curvatures), because both are unstable.', ''),
# ---- RANSAC and homography
(F + 'ransac', 'ransac', 'RANSAC is used in homography estimation because some feature matches are wrong (outliers); it repeatedly fits a model to a random minimal sample (4 pairs) and keeps the model that most matches agree with, so outliers do not distort the result.', ''),
(F + 'perspective', 'perspective', 'A homography is a 3x3 matrix with 8 degrees of freedom (defined up to scale), so at least 4 point correspondences, no three collinear, are needed to compute it.', ''),
(F + 'affine', 'affine', 'An affine transform has 6 unknowns (2x3 matrix), so 3 non-collinear point pairs determine it; it preserves parallel lines and ratios along a line but not angles or lengths.', ''),
(F + 'perspective', 'perspective', 'Affine versus perspective: affine keeps parallel lines parallel (2x3 matrix, 3 point pairs); perspective/projective can make parallel lines converge (3x3 matrix, 4 point pairs), modelling a change of camera viewpoint.', ''),
(F + 'affine', 'affine', 'Transformations are composed by multiplying their 3x3 homogeneous matrices; the order matters because matrix multiplication is not commutative (the rightmost matrix is applied first).', ''),
(F + 'affine', 'affine', 'Homogeneous coordinates add a third coordinate (x, y, 1) so translation can be written as a matrix multiplication and combined with rotation and scaling in one matrix.', ''),
(F + 'panorama', 'panorama', 'Panorama stitching steps: detect keypoints and descriptors in overlapping images, match them with a ratio test, estimate a homography with RANSAC, warp one image into the other\'s frame, then blend the overlap.', ''),
# ---- features
(F + 'hog', 'hog', 'HOG describes shape by the distribution of gradient directions: the window is split into cells, each cell gets a histogram of gradient orientations weighted by magnitude, and blocks of cells are normalized for illumination robustness.', ''),
(F + 'lbp', 'lbp', 'LBP describes texture by comparing each pixel with its neighbours: each neighbour that is greater than or equal to the center gives a 1, the bits form a code, and the histogram of codes over a region is the texture descriptor.', ''),
(F + 'lbp', 'lbp', 'LBP is robust to uniform (monotonic) brightness changes because it depends only on whether neighbours are brighter or darker than the center, not on their absolute values.', ''),
(F + 'corners', 'corners', 'Harris detects corners where shifting a small window in any direction changes the intensity strongly; along an edge the change is large in only one direction, in a flat region in none.', ''),
(F + 'gradients', 'gradients', 'Sobel results are computed in a float depth such as CV_64F because derivatives can be negative; with uint8 the negative edges (bright-to-dark) would be clipped to zero.', ''),
# ---- enhancement
(F + 'equalization', 'equalization', 'Histogram equalization spreads the most frequent intensities over the full 0-255 range using the cumulative histogram, which increases global contrast.', ''),
(F + 'equalization', 'equalization', 'CLAHE equalizes small tiles separately and clips each histogram at a limit before equalizing, so it improves local contrast without over-amplifying noise as global equalization can.', ''),
(F + 'gamma', 'gamma', 'Gamma correction maps each value as 255 * (I / 255) ** gamma; gamma below 1 brightens dark regions and gamma above 1 darkens bright regions.', ''),
# ---- change detection
(F + 'background', 'background', 'Frame differencing detects motion by subtracting a reference (background) frame from the current frame and thresholding the absolute difference; it needs a static camera because camera motion changes every pixel.', ''),
# ---- wavelets
(F + 'wavelets', 'wavelets', 'Wavelet denoising splits a signal into approximation (smooth trend) and detail coefficients, shrinks small detail coefficients that mostly hold noise, and reconstructs; large deviations from the denoised signal can then be flagged as anomalies.', ''),
]
