"""Manual content the OCR pass missed: figure labels, code read from page scans, page headers.

Each entry was checked against the page image (extracted/page_images/<source>_pNNN.jpg).
"""

FACTS = [
# Page headers / metadata, present on every manual page.
('lab_01_manual:p1', 'features', 'All lab manuals are from the National University of Computer & Emerging Sciences (NUCES / FAST NUCES) Karachi, Department of Artificial Intelligence.', ''),
('lab_01_manual:p1', 'features', 'The course is Computer Vision Lab, course code AI-4002; the instructor is Talha Shahid.', ''),
('lab_01_manual:p1', 'features', 'Lab titles: Lab 01 Introduction to Computer Vision and applications; Lab 03 Geometric Transformation; Lab 04 Feature Extraction; Lab 05 Image Segmentation; Lab Manual 06 Wavelets, Boundary Detection, Hough Transform and SIFT.', ''),
('lab_05_manual:p1', 'segmentation', 'The main topic of Lab 05 is Image Segmentation.', ''),
# Lab 05 figures (p3, p4).
('lab_05_manual:p3', 'segmentation', 'In the Lab 05 landscape figure, preprocessing and feature extraction uses pixel values, edge detection, texture analysis and color similarity.', ''),
('lab_05_manual:p3', 'segmentation', 'In the Lab 05 landscape figure, segmentation algorithms shown are region growing, clustering (K-means), watershed and a deep learning model (CNN).', ''),
('lab_05_manual:p3', 'segmentation', 'The landscape example output segmented region map has segment 1 sky, segment 2 mountains and segment 3 water.', ''),
('lab_05_manual:p4', 'segmentation', 'The portrait example segments the image into hair region, face region and clothing region only; no background region is labeled.', ''),
('lab_05_manual:p4', 'segmentation', 'The street scene example segments the image into segment A car, segment B tree and segment C person.', ''),
('lab_05_manual:p4', 'segmentation', 'Segmentation is the process of dividing an image into meaningful, separate parts; the result is distinct, non-overlapping regions representing meaningful objects.', ''),
('lab_05_manual:p3', 'segmentation', 'Visual characteristics used to group pixels in segmentation are color, texture and intensity; file size and metadata are not visual characteristics.', ''),
# Lab 05 code read from the scan (p11, p13, p14).
('lab_05_manual:p13', 'watershed', 'The watershed code creates markers with cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU).', ''),
('lab_05_manual:p13', 'watershed', 'The watershed code marks boundaries with image[markers == -1] = [255, 0, 0], commented as red.', 'In a BGR image [255, 0, 0] is blue; it shows red only because plt.imshow reads the array as RGB (correction C23).'),
('lab_05_manual:p13', 'watershed', '_, markers = cv2.connectedComponents(sure_fg) labels the sure foreground; the second return value is the markers (labels) image.', ''),
('lab_05_manual:p12', 'watershed', 'Watershed post-processing optionally applies morphological operations such as erosion and dilation to refine the segmentation.', ''),
('lab_05_manual:p12', 'watershed', 'Watershed preprocessing converts the input image to grayscale and may apply noise reduction or contrast enhancement.', ''),
('lab_05_manual:p12', 'watershed', 'Watershed gradient calculation uses Sobel or Scharr filters to highlight object boundaries.', ''),
('lab_05_manual:p11', 'region', 'In region growing, a pixel whose intensity difference from the seed is below the threshold is added to the segmented region (mask set to 255) and its neighbours are pushed on the stack.', ''),
('lab_05_manual:p11', 'region', 'The region growing code initializes seed_point = (10, 10), uses threshold = 50 and creates the mask with np.zeros_like(image, dtype=np.uint8).', ''),
('lab_05_manual:p11', 'region', 'The region growing code pushes the 4-connected neighbours (x+1, y), (x-1, y), (x, y+1), (x, y-1) and uses a stack for pixel traversal.', ''),
('lab_05_manual:p14', 'kmeans', 'Clustering post-processing can merge or split clusters based on specific criteria to refine the segmentation.', ''),
('lab_05_manual:p14', 'kmeans', 'In the K-means update step each centroid is recalculated as the mean of the pixels assigned to its cluster (not the median or mode).', ''),
# Lab 01 table rows missing from OCR (p7-p8).
('lab_01_manual:p8', 'features', 'Hugging Face is known for pretrained NLP and CV models (transformers), integrating with PyTorch and TensorFlow.', ''),
('lab_01_manual:p7', 'features', 'OpenCV is the library mainly used for general computer vision tasks; Keras is a high-level neural network API, Caffe a CNN deep learning framework and Detectron2 an object detection library.', ''),
# Lab 03 p6.
('lab_03_manual:p6', 'pixels', 'The image coordinate system matters because transformations such as rotation and cropping are defined relative to it; OpenCV accounts for the top-left origin when defining positive rotation.', ''),
]

TRIPLES = [
(['Lab 05'], 'main topic', 'Image Segmentation', 'lab_05_manual:p1'),
(['university', 'lab manual'], 'from university', 'NUCES Karachi', 'lab_01_manual:p1'),
(['course code'], 'course code', 'AI-4002', 'lab_01_manual:p1'),
(['landscape example'], 'segmented region map includes', 'sky, mountains, water', 'lab_05_manual:p3'),
(['street scene example', 'street scene'], 'segments include', 'car, tree, person', 'lab_05_manual:p4'),
(['portrait example'], 'regions shown', 'hair region, face region, clothing region', 'lab_05_manual:p4'),
(['preprocessing and feature extraction'], 'includes in segmentation', 'pixel values, edge detection, texture analysis, color similarity', 'lab_05_manual:p3'),
(['watershed code', 'markers'], 'threshold type used for markers', 'THRESH_BINARY_INV + THRESH_OTSU', 'lab_05_manual:p13'),
(['watershed post-processing', 'post-processing'], 'watershed post-processing uses', 'morphological operations', 'lab_05_manual:p12'),
(['clustering', 'post-processing in clustering'], 'post-processing can involve', 'merging or splitting clusters', 'lab_05_manual:p14'),
(['update step'], 'recalculates centroids as', 'mean of pixels in each cluster', 'lab_05_manual:p14'),
(['Hugging Face'], 'known for', 'pretrained NLP and CV models', 'lab_01_manual:p8'),
(['image coordinate system'], 'important because', 'it affects how transformations like rotation and cropping are applied', 'lab_03_manual:p6'),
(['Sobel'], 'number of convolution kernels', 'two', 'lab_manual_06:p6'),
(['Sobel kernels', 'Sobel'], 'kernel size', '3x3', 'lab_manual_06:p6'),
(['Non-Maximum Suppression', 'NMS'], 'does in Canny', 'thins edges to 1-pixel width', 'lab_manual_06:p12'),
(['line', 'line in Cartesian coordinates'], 'represented as in Hough', 'y = mx + b', 'lab_manual_06:p17'),
(['circle', 'circle in Cartesian coordinates'], 'represented as in Hough', '(x - a)^2 + (y - b)^2 = r^2', 'lab_manual_06:p18'),
(['scale-space extrema detection'], 'finds', 'local extrema in scale space', 'lab_manual_06:p18'),
(['outlier rejection'], 'uses in SIFT matching', 'RANSAC', 'lab_manual_06:p21'),
(['Gaussian blur', 'GaussianBlur'], 'kernel size used in the lab', '(5, 5)', 'lab_manual_06:p13'),
(['Gaussian blur', 'sigmaX'], 'sigmaX value used in the lab', '1.4', 'lab_manual_06:p13'),
(['cv2.Canny', 'Canny'], 'lower and upper thresholds used in the lab', '50, 150', 'lab_manual_06:p13'),
(['cv2.Laplacian', 'Laplacian'], 'depth used in the lab', 'cv2.CV_64F', 'lab_manual_06:p15'),
(['cv2.convertScaleAbs', 'convertScaleAbs'], 'does', 'converts to absolute values and scales to 8-bit', 'opencv_docs:convertScaleAbs'),
(['COLOR_BGR2RGB'], 'does in cv2.cvtColor', 'converts BGR to RGB for Matplotlib', 'lab_01_manual:p14'),
(['grayscale conversion', 'COLOR_BGR2GRAY'], 'code used in OpenCV', 'cv2.COLOR_BGR2GRAY', 'lab_01_manual:p15'),
(['imread', 'cv2.imread()'], 'flag for grayscale', 'IMREAD_GRAYSCALE', 'lab_01_manual:p24'),
(['pixels between low and high threshold', 'between low and high threshold'], 'are in Canny', 'weak edge points', 'lab_04_manual:p22'),
(['HED histogram', 'HED'], 'histogram bins in code', '8', 'lab_04_manual:p11'),
]
