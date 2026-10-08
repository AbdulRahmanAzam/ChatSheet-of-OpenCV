"""Atomic, manual-grounded facts for exam-style (MCQ / short answer) questions.

Each fact restates one claim from a manual page in clean words (OCR noise removed).
`note` records where the manual's claim is imprecise; exam answers follow the
manual, while `note` gives the technically precise view (see records/corrections.jsonl).

Format: (source page ref, topic id, statement, note)
"""

FACTS = [
# ---- Lab 04 manual: introduction to feature extraction (p4-p6)
('lab_04_manual:p4', 'features', 'Feature extraction selects and transforms relevant information from raw visual data (images or videos) into a more compact and meaningful representation.', ''),
('lab_04_manual:p4', 'features', 'The compact feature representation is used for tasks such as object recognition, image classification and facial recognition.', ''),
('lab_04_manual:p4', 'features', 'Raw visual data contains a vast amount of information including pixel values, colors and textures.', ''),
('lab_04_manual:p4', 'features', 'Raw visual data is often too complex and high-dimensional for direct analysis and interpretation by algorithms.', ''),
('lab_04_manual:p4', 'features', 'Feature extraction reduces the dimensionality of the data, enhances relevant information and simplifies computational requirements.', ''),
('lab_04_manual:p4', 'features', 'Features are distinctive characteristics that should capture essential information about objects, patterns or structures while discarding irrelevant details.', ''),
('lab_04_manual:p5', 'features', 'Local features are extracted from specific regions of an image such as keypoints, corners or small patches.', ''),
('lab_04_manual:p5', 'features', 'Local features are often used for image matching and object detection.', ''),
('lab_04_manual:p5', 'features', 'Global features are computed over the entire image and capture its overall characteristics.', ''),
('lab_04_manual:p5', 'features', 'Examples of global features are color histograms, texture descriptors and image moments.', ''),
('lab_04_manual:p5', 'features', 'Histogram-based methods: histograms of color or texture distributions capture statistical information about the image.', ''),
('lab_04_manual:p5', 'features', 'Filter-based methods: filters like Gabor filters or Haar wavelets identify edges, textures or specific patterns.', ''),
('lab_04_manual:p5', 'features', 'Deep learning features: convolutional neural networks (CNNs) learn features directly from the data through the network layers.', ''),
('lab_04_manual:p5', 'features', 'Applications of feature extraction: object recognition, face recognition, gesture recognition, medical imaging and autonomous vehicles.', ''),
('lab_04_manual:p5', 'features', 'Face recognition extracts facial features like eyes, nose and mouth for identity verification.', ''),
('lab_04_manual:p5', 'features', 'Feature extraction challenges are variations in lighting, scale, orientation and noise in real-world data.', ''),
('lab_04_manual:p6', 'features', 'Choosing appropriate feature extraction techniques and parameters is critical for successful computer vision applications.', ''),
('lab_04_manual:p6', 'features', 'Extracted features are often used with matching or learning algorithms for object recognition or image classification.', ''),
# ---- HOG (p6-p7)
('lab_04_manual:p6', 'hog', 'HOG stands for Histogram of Oriented Gradients.', ''),
('lab_04_manual:p6', 'hog', 'HOG is particularly useful for object detection and pedestrian detection.', ''),
('lab_04_manual:p6', 'hog', 'HOG captures information about local gradient or edge patterns, the distribution of gradient directions in an image.', ''),
('lab_04_manual:p6', 'hog', 'HOG divides the image into small cells and computes gradient magnitude and orientation for each pixel in the cells.', ''),
('lab_04_manual:p6', 'hog', 'HOG step 1 image preprocessing: convert the input image to grayscale to simplify gradient calculation.', ''),
('lab_04_manual:p6', 'hog', 'HOG gradient computation typically uses the Sobel operator.', ''),
('lab_04_manual:p6', 'hog', 'HOG cell division: the image is divided into small non-overlapping cells, typical cell size 8x8 pixels.', ''),
('lab_04_manual:p6', 'hog', 'In HOG each cell gets a histogram of gradient orientations; the histogram bins represent different orientation ranges.', ''),
('lab_04_manual:p7', 'hog', 'HOG block normalization groups neighboring cells into blocks (typical block 2x2 cells) and normalizes them to reduce the effects of lighting variations.', ''),
('lab_04_manual:p7', 'hog', 'The final HOG feature vector is formed by concatenating the normalized histograms from the blocks.', ''),
('lab_04_manual:p7', 'hog', 'The HOG code uses skimage.feature.hog (from skimage.feature import hog), not an OpenCV function.', 'OpenCV also provides cv2.HOGDescriptor, but the manual code uses skimage.feature.hog.'),
('lab_04_manual:p7', 'hog', 'In skimage hog, pixels_per_cell=(8,8) sets the cell size and cells_per_block=(2,2) sets the number of cells per block; orientations sets the number of bins.', ''),
('lab_04_manual:p7', 'hog', 'The HOG visualization image is rescaled with skimage exposure.rescale_intensity for display.', ''),
# ---- LBP (p7-p9)
('lab_04_manual:p7', 'lbp', 'LBP stands for Local Binary Pattern, a texture feature extraction technique.', ''),
('lab_04_manual:p7', 'lbp', 'LBP captures local patterns by comparing the intensity of a pixel with its neighboring pixels.', ''),
('lab_04_manual:p7', 'lbp', 'LBP is particularly useful for texture classification and face recognition.', ''),
('lab_04_manual:p8', 'lbp', 'LBP operates on grayscale images and defines a circular neighborhood around each pixel.', ''),
('lab_04_manual:p8', 'lbp', 'In LBP, if the neighbor intensity is greater than or equal to the center pixel intensity, the neighbor is assigned 1.', ''),
('lab_04_manual:p8', 'lbp', 'In LBP, if the neighbor intensity is less than the center pixel intensity, the neighbor is assigned 0.', ''),
('lab_04_manual:p8', 'lbp', 'LBP concatenates the binary comparison values to form an LBP pattern, a binary number.', ''),
('lab_04_manual:p8', 'lbp', 'The LBP histogram counts the occurrences of each LBP pattern in the image.', ''),
('lab_04_manual:p8', 'lbp', 'The LBP histogram serves as the feature vector for texture analysis or classification.', ''),
('lab_04_manual:p8', 'lbp', 'The LBP code uses skimage.feature.local_binary_pattern (from skimage import feature).', ''),
('lab_04_manual:p8', 'lbp', 'In the LBP code radius = 1 is the radius of the circular neighborhood and n_points = 8 * radius is the number of neighboring pixels to consider.', ''),
('lab_04_manual:p9', 'lbp', "In the LBP code method='uniform' uses uniform patterns, which reduces the number of patterns (n_points + 2 histogram bins).", ''),
('lab_04_manual:p9', 'lbp', 'The LBP histogram is normalized by dividing by its sum (hist.sum() + 1e-6).', ''),
# ---- Histogram of color (p9)
('lab_04_manual:p9', 'histogram', 'Histogram of Color captures the distribution of color intensities in an image.', ''),
('lab_04_manual:p9', 'histogram', 'Histogram of Color divides the color space (RGB or HSV) into bins and counts the number of pixels in each bin.', ''),
('lab_04_manual:p5', 'histogram', 'Histogram of Color (a color histogram) is an example of a global feature because it is computed over the entire image.', ''),
# ---- HED (p9-p11)
('lab_04_manual:p9', 'hed', 'HED stands for Histogram of Edge Directions.', ''),
('lab_04_manual:p9', 'hed', 'HED captures the distribution of edge orientations and the dominant edge directions in an image.', ''),
('lab_04_manual:p9', 'hed', 'HED is useful for texture analysis, object recognition and image segmentation.', ''),
('lab_04_manual:p10', 'hed', 'HED preprocessing converts to grayscale and optionally applies Gaussian smoothing to reduce noise.', ''),
('lab_04_manual:p10', 'hed', 'HED edge detection can use the Canny edge detector, the Sobel operator or the Scharr operator, producing an edge map.', ''),
('lab_04_manual:p10', 'hed', 'In HED the orientation (angle) of the gradient at each pixel is calculated using the arctangent function.', ''),
('lab_04_manual:p10', 'hed', 'In HED orientation binning, 360 degrees is divided into 8 bins for octagonal directions.', ''),
('lab_04_manual:p10', 'hed', 'The HED feature vector is the histogram of edge directions, counting pixels in each orientation bin.', ''),
('lab_04_manual:p11', 'hed', 'In the HED code gradient orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi, with y first then x.', ''),
('lab_04_manual:p11', 'hed', 'The HED code builds the histogram with np.histogram(gradient_orientation, bins=8, range=(0, 360)): 8 bins over the range 0 to 360.', 'arctan2 returns -180..180 degrees, so negative angles fall outside range (0, 360) unless wrapped (correction C11).'),
('lab_04_manual:p11', 'hed', 'The HED code computes gradient_x and gradient_y with cv2.Sobel(..., cv2.CV_64F, ...) and magnitude np.sqrt(gradient_x**2 + gradient_y**2).', ''),
# ---- HIG (p11-p13)
('lab_04_manual:p11', 'hig', 'HIG stands for Histogram of Intensity Gradients.', ''),
('lab_04_manual:p11', 'hig', 'HIG captures the distribution of intensity gradients in an image.', ''),
('lab_04_manual:p11', 'hig', 'HIG is useful for object recognition, texture analysis and image classification.', ''),
('lab_04_manual:p12', 'hig', 'HIG computes the gradient with the Sobel operator, Scharr operator or other gradient methods, separately for horizontal dx and vertical dy.', ''),
('lab_04_manual:p12', 'hig', 'Gradient magnitude formula: G = sqrt(dx^2 + dy^2).', ''),
('lab_04_manual:p12', 'hig', 'Gradient orientation formula: theta = atan2(dy, dx), with dy first then dx.', ''),
('lab_04_manual:p12', 'hig', 'In HIG the range of gradient orientations is typically 0 to 360 degrees, divided into bins.', ''),
('lab_04_manual:p12', 'hig', 'The HIG histogram can optionally be normalized to make it scale-invariant.', 'Normalization removes dependence on pixel count, not true spatial scale invariance (correction C13).'),
('lab_04_manual:p12', 'hig', 'The HIG feature vector is the histogram of gradient orientations, capturing intensity gradient information.', ''),
('lab_04_manual:p12', 'hig', 'HIG is similar to HOG but is usually computed globally over the whole image instead of local cells and blocks.', 'Inference from the manual: HIG has no cell/block step.'),
('lab_04_manual:p13', 'hig', 'The HIG code uses np.histogram(orientation, bins=9, range=(0, 180)) and normalizes with histogram / histogram.sum().', 'The text says 0 to 360 degrees; the code example uses 9 bins over (0, 180).'),
# ---- Texture energy and contrast (p13-p15)
('lab_04_manual:p13', 'texture', 'Texture energy and contrast histograms provide information about the variation and contrast of textures.', ''),
('lab_04_manual:p13', 'texture', 'Texture energy is the sum of squared pixel intensities within a local neighborhood: E = sum(pixel values^2).', ''),
('lab_04_manual:p13', 'texture', 'Texture energy measures uniformity or smoothness; the manual says higher energy indicates a more complex, less uniform texture.', 'A uniform bright patch also has high energy; energy alone mixes brightness and variation (correction C14).'),
('lab_04_manual:p13', 'texture', 'Texture contrast is the standard deviation of pixel intensities in the neighborhood: C = sqrt(Var(pixel values)).', ''),
('lab_04_manual:p13', 'texture', 'Higher texture contrast values indicate greater variation in texture.', ''),
('lab_04_manual:p14', 'texture', 'Common neighborhood (window) sizes for texture analysis are 3x3 or 5x5.', ''),
('lab_04_manual:p14', 'texture', 'The texture energy histogram captures the distribution of texture energy values.', ''),
('lab_04_manual:p14', 'texture', 'The texture contrast histogram captures the distribution of contrast values.', ''),
('lab_04_manual:p14', 'texture', 'Texture energy and contrast histograms serve as feature vectors.', ''),
('lab_04_manual:p14', 'texture', 'Texture energy and contrast features are used for texture classification and segmentation.', ''),
('lab_04_manual:p14', 'texture', 'The texture code computes energy with cv2.filter2D(image**2, -1, np.ones((3,3))), which sums squared pixel values in the neighborhood.', 'Convert to float before squaring; uint8 squares overflow (correction C15).'),
('lab_04_manual:p15', 'texture', 'The texture code creates histograms with np.histogram (bins=256) and normalizing the histograms is optional.', ''),
('lab_04_manual:p15', 'texture', 'Task: texture analysis for material classification of wood, metal and fabric using at least two texture features.', ''),
# ---- Filtering and convolution (p16-p19)
('lab_04_manual:p16', 'kernel', 'Filtering and convolution are used for noise reduction, edge detection, feature extraction and image enhancement.', ''),
('lab_04_manual:p16', 'kernel', 'Filtering applies a filter or kernel, a small matrix of numbers, to an input image.', ''),
('lab_04_manual:p16', 'kernel', 'Each element of the filter represents a weighted contribution to the new pixel value.', ''),
('lab_04_manual:p16', 'kernel', 'Convolution slides the filter over the image, element-wise multiplies the filter and the image region, and sums the results (a dot product).', ''),
('lab_04_manual:p16', 'kernel', 'The result of convolution is a new image called the output or convolved image.', ''),
('lab_04_manual:p16', 'box', 'Box blur replaces each pixel with the average of its neighboring pixels within a square kernel; useful for noise reduction and smoothing.', ''),
('lab_04_manual:p16', 'box', 'The box blur kernel in the manual is np.ones((3,3), dtype=np.float32) / 9 applied with cv2.filter2D(image, -1, kernel).', ''),
('lab_04_manual:p17', 'gaussian', 'Gaussian blur uses a Gaussian-shaped kernel and gives smoother results than box blur; often used for noise reduction.', ''),
('lab_04_manual:p17', 'gaussian', 'The manual builds the Gaussian kernel with cv2.getGaussianKernel(kernel_size, sigma) and applies it with cv2.filter2D.', 'getGaussianKernel returns a 1-D column; use v @ v.T for a 2-D blur (correction C16).'),
('lab_04_manual:p17', 'gradients', 'Sobel and Scharr operators detect edges by emphasizing rapid changes in pixel values.', ''),
('lab_04_manual:p17', 'kernel', 'Embossing creates a 3D effect by emphasizing differences in neighboring pixel values; used for artistic or stylized effects.', ''),
('lab_04_manual:p17', 'padding', 'Standard convolution is also known as full convolution and is the most common type; it centers the kernel over each pixel.', 'In signal processing "full" means output larger than input; the manual example actually produces same-size output (correction C17).'),
('lab_04_manual:p18', 'padding', 'Valid convolution (no padding) only processes pixels where the kernel fully overlaps the image and produces an output smaller than the input.', ''),
('lab_04_manual:p18', 'padding', 'Same convolution adds zero-padding so the output image has the same dimensions as the input, preventing information loss at the boundaries.', ''),
('lab_04_manual:p19', 'padding', 'Valid convolution with strides skips some pixels based on the specified stride, giving an output smaller than the input determined by the stride.', 'cv2.filter2D has no strides argument; subsample after filtering (correction C18).'),
# ---- Edge detection (p19-p23)
('lab_04_manual:p19', 'canny', 'Edge detection identifies boundaries within an image, transitions from one object or region to another.', ''),
('lab_04_manual:p19', 'canny', 'Edges correspond to changes in color, intensity or texture.', ''),
('lab_04_manual:p19', 'canny', 'Edge detection is a preprocessing step that reduces the amount of data while highlighting essential features.', ''),
('lab_04_manual:p20', 'gradients', 'Gradient-based edge detection calculates the gradient (rate of change) of pixel intensities.', ''),
('lab_04_manual:p20', 'gradients', 'A high gradient magnitude indicates an edge is likely present; edges typically occur where gradient magnitude is high.', ''),
('lab_04_manual:p20', 'gradients', 'Common gradient-based operators are Sobel, Prewitt and Scharr.', ''),
('lab_04_manual:p20', 'canny', 'Canny is a multi-stage edge detector known for its accuracy and noise reduction capabilities.', ''),
('lab_04_manual:p20', 'laplacian', 'LoG stands for Laplacian of Gaussian; it combines Gaussian smoothing and Laplacian edge detection and highlights zero-crossings of the second derivative.', ''),
('lab_04_manual:p20', 'gradients', 'Sobel and Scharr are gradient-based edge detectors approximating the image gradient in horizontal and vertical directions.', ''),
('lab_04_manual:p20', 'scharr', 'The Prewitt operator is a gradient-based method using 3x3 kernels for horizontal and vertical gradients.', ''),
('lab_04_manual:p21', 'laplacian', 'The Marr-Hildreth edge detector uses the LoG operator after Gaussian smoothing and locates edges at zero-crossings.', ''),
('lab_04_manual:p21', 'gradients', 'The manual describes the Sobel-Feldman operator as a variation of Sobel emphasizing diagonal edges in addition to horizontal and vertical edges.', 'Sobel-Feldman is simply the full name of the Sobel operator (correction C20).'),
('lab_04_manual:p21', 'canny', 'CNN-based edge detection learns complex edge patterns and adapts to various image domains.', ''),
('lab_04_manual:p21', 'canny', 'The Canny edge detector was developed by John F. Canny in 1986.', ''),
('lab_04_manual:p22', 'canny', 'Canny step 1 (first step): Gaussian smoothing reduces noise by convolving a Gaussian kernel with the image.', ''),
('lab_04_manual:p22', 'canny', 'Canny step 2 (second step): gradient calculation with two 3x3 Sobel kernels gives gradient magnitude and direction.', ''),
('lab_04_manual:p22', 'canny', 'Canny step 3 (third step): non-maximum suppression keeps local maxima along the gradient direction and thins the edges.', ''),
('lab_04_manual:p22', 'canny', 'Canny step 4 (final step): edge tracking by hysteresis uses two thresholds, a high threshold and a low threshold.', ''),
('lab_04_manual:p22', 'canny', 'In Canny, pixels with gradient magnitude above the high threshold are strong edge points.', ''),
('lab_04_manual:p22', 'canny', 'In Canny, pixels between the low and high threshold are weak edge points.', ''),
('lab_04_manual:p22', 'canny', 'In Canny, weak edge points are kept only if connected to strong edge points; pixels below the low threshold are rejected as non-edge points.', ''),
('lab_04_manual:p23', 'canny', 'The Canny parameter sigma sets the Gaussian kernel size and smoothing; a larger sigma gives a smoother image but loses fine details.', ''),
('lab_04_manual:p23', 'canny', 'In OpenCV cv2.Canny, threshold1 is the low threshold (lower hysteresis threshold).', ''),
('lab_04_manual:p23', 'canny', 'In OpenCV cv2.Canny, threshold2 is the high threshold (upper hysteresis threshold).', ''),
('lab_04_manual:p23', 'canny', 'The Canny code is cv2.Canny(image, threshold1=100, threshold2=200).', ''),
('lab_04_manual:p23', 'canny', 'Choosing Canny thresholds trades off edge detection sensitivity against noise suppression.', ''),
]

# Other manuals live in their own files.
from facts_lab01 import FACTS as _LAB01
from facts_lab03 import FACTS as _LAB03
from facts_lab05 import FACTS as _LAB05
from facts_lab06 import FACTS as _LAB06
FACTS = FACTS + _LAB01 + _LAB03 + _LAB05 + _LAB06

# OpenCV parameter docs / CV fundamentals, plus values read from the installed library.
from facts_opencv_api import FACTS as _API
import facts_introspect as _intro
_INTRO_FACTS, INTRO_TRIPLES = _intro.build()
FACTS = FACTS + _API + _INTRO_FACTS

# Content OCR missed (figures, code from scans, page headers).
from facts_gaps import FACTS as _GAPS
FACTS = FACTS + _GAPS

# Conceptual why/how explanations (CV fundamentals, marked as not from the manual).
from facts_concepts import FACTS as _CONCEPTS
FACTS = FACTS + _CONCEPTS
