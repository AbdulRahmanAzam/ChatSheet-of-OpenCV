# Manual code inventory

38 identified example groups are mapped below. Full-page image links preserve original code exactly as visually printed. Raw OCR is included in records/manual_examples.jsonl and is NOT a verified verbatim transcription. Corrected equivalents change unsafe/incorrect examples and may change display styling or optional dependencies.

## M01-14 — Read/display color

Corrected implementation: `manual_examples.basic_examples`; select `color`.



- [lab_01_manual:p14](extracted/page_images/lab_01_manual_p014.jpg)

## M01-15 — Grayscale display

Corrected implementation: `manual_examples.basic_examples`; select `gray`.



- [lab_01_manual:p15](extracted/page_images/lab_01_manual_p015.jpg)

## M01-16 — Resize half size

Corrected implementation: `manual_examples.basic_examples`; select `half_size`.



- [lab_01_manual:p16](extracted/page_images/lab_01_manual_p016.jpg)

## M01-17 — Gaussian blur31

Corrected implementation: `manual_examples.basic_examples`; select `gaussian31`.



- [lab_01_manual:p17](extracted/page_images/lab_01_manual_p017.jpg)

## M01-18 — Left half crop

Corrected implementation: `manual_examples.basic_examples`; select `left_crop`.



- [lab_01_manual:p18](extracted/page_images/lab_01_manual_p018.jpg)

## M01-19 — Text overlay

Corrected implementation: `manual_examples.basic_examples`; select `text`.



- [lab_01_manual:p19](extracted/page_images/lab_01_manual_p019.jpg)

## M01-21 — Threshold100/max200

Corrected implementation: `manual_examples.basic_examples`; select `threshold100_max200`.



- [lab_01_manual:p21](extracted/page_images/lab_01_manual_p021.jpg)

## M01-22 — Rotation45 same canvas

Corrected implementation: `manual_examples.basic_examples`; select `rotate_same_canvas`.



- [lab_01_manual:p22](extracted/page_images/lab_01_manual_p022.jpg)

## M01-23 — Saturating addition

Corrected implementation: `manual_examples.basic_examples`; select `saturating_add`.



- [lab_01_manual:p23](extracted/page_images/lab_01_manual_p023.jpg)

## M01-24 — Histogram equalization

Corrected implementation: `manual_examples.basic_examples`; select `equalized`.



- [lab_01_manual:p24](extracted/page_images/lab_01_manual_p024.jpg)

## M01-20 — Pandas RGB pixel summary

Corrected implementation: `lab01.task09_optional_pandas`; select ``.

Pandas is optional and outside the core dependency scope. NumPy equivalent is task09_rgb_statistics.

- [lab_01_manual:p20](extracted/page_images/lab_01_manual_p020.jpg)

## M04-HOG — HOG extraction and visualization

Corrected implementation: `manual_examples.feature_examples`; select `hog`.

OpenCV HOG descriptor replacement; original skimage HOG visualization not reproduced bit-for-bit.

- [lab_04_manual:p7](extracted/page_images/lab_04_manual_p007.jpg)

## M04-LBP — Uniform local binary pattern

Corrected implementation: `manual_examples.feature_examples`; select `lbp_histogram`.

Circular8-neighbor replacement; border/sampling differs from scikit-image.

- [lab_04_manual:p8](extracted/page_images/lab_04_manual_p008.jpg)
- [lab_04_manual:p9](extracted/page_images/lab_04_manual_p009.jpg)

## M04-HED — Histogram of edge directions

Corrected implementation: `manual_examples.feature_examples`; select `hed`.

Fix angle wrapping and apply edge mask.

- [lab_04_manual:p10](extracted/page_images/lab_04_manual_p010.jpg)
- [lab_04_manual:p11](extracted/page_images/lab_04_manual_p011.jpg)

## M04-HIG — Histogram of intensity gradients

Corrected implementation: `manual_examples.feature_examples`; select `hig`.

Fix angle wrapping; explicitly magnitude-weighted.

- [lab_04_manual:p12](extracted/page_images/lab_04_manual_p012.jpg)
- [lab_04_manual:p13](extracted/page_images/lab_04_manual_p013.jpg)

## M04-TEXTURE — Texture energy and contrast

Corrected implementation: `manual_examples.feature_examples`; select `energy;contrast`.

Float intermediates; implement stated standard-deviation contrast.

- [lab_04_manual:p14](extracted/page_images/lab_04_manual_p014.jpg)
- [lab_04_manual:p15](extracted/page_images/lab_04_manual_p015.jpg)

## M04-BOX — Box smoothing

Corrected implementation: `manual_examples.filtering_examples`; select `box`.



- [lab_04_manual:p16](extracted/page_images/lab_04_manual_p016.jpg)

## M04-GAUSS — Gaussian kernel filtering

Corrected implementation: `manual_examples.filtering_examples`; select `gaussian2d;gaussian_separable`.

Correct1D column to2D filtering.

- [lab_04_manual:p17](extracted/page_images/lab_04_manual_p017.jpg)

## M04-DERIV — Sobel and Scharr

Corrected implementation: `manual_examples.filtering_examples`; select `sobel_x;sobel_y;scharr_x;scharr_y`.



- [lab_04_manual:p17](extracted/page_images/lab_04_manual_p017.jpg)

## M04-EMBOSS — Emboss kernel

Corrected implementation: `manual_examples.filtering_examples`; select `emboss_signed`.

Keep signed response for analysis.

- [lab_04_manual:p17](extracted/page_images/lab_04_manual_p017.jpg)

## M04-STANDARD — Standard filtering

Corrected implementation: `manual_examples.filtering_examples`; select `correlation_same;convolution_full`.

Distinguish correlation from full convolution.

- [lab_04_manual:p17](extracted/page_images/lab_04_manual_p017.jpg)
- [lab_04_manual:p18](extracted/page_images/lab_04_manual_p018.jpg)

## M04-VALID — Valid convolution

Corrected implementation: `manual_examples.filtering_examples`; select `convolution_valid`.

Crop valid support; borderconstant alone is insufficient.

- [lab_04_manual:p18](extracted/page_images/lab_04_manual_p018.jpg)

## M04-SAME — Same zero-padded convolution

Corrected implementation: `manual_examples.filtering_examples`; select `convolution_same_zero`.

Use zero padding, not reflected borders.

- [lab_04_manual:p18](extracted/page_images/lab_04_manual_p018.jpg)
- [lab_04_manual:p19](extracted/page_images/lab_04_manual_p019.jpg)

## M04-STRIDE — Strided valid convolution

Corrected implementation: `manual_examples.filtering_examples`; select `convolution_stride2`.

Replace unsupported strides keyword with explicit subsampling.

- [lab_04_manual:p19](extracted/page_images/lab_04_manual_p019.jpg)

## M04-GRADIENT — Gradient magnitude threshold

Corrected implementation: `cv_core.gradients`; select `magnitude`.

Apply (magnitude>100).astype(np.uint8)*255 for original threshold demonstration.

- [lab_04_manual:p21](extracted/page_images/lab_04_manual_p021.jpg)

## M04-CANNY — Canny edge example

Corrected implementation: `manual_examples.segmentation_examples`; select `canny`.

Explicit Gaussian preprocessing documented.

- [lab_04_manual:p23](extracted/page_images/lab_04_manual_p023.jpg)

## M05-GLOBAL — Global threshold

Corrected implementation: `manual_examples.segmentation_examples`; select `global`.



- [lab_05_manual:p8](extracted/page_images/lab_05_manual_p008.jpg)

## M05-ADAPT — Adaptive threshold

Corrected implementation: `manual_examples.segmentation_examples`; select `adaptive`.



- [lab_05_manual:p8](extracted/page_images/lab_05_manual_p008.jpg)
- [lab_05_manual:p9](extracted/page_images/lab_05_manual_p009.jpg)

## M05-OTSU — Otsu threshold

Corrected implementation: `manual_examples.segmentation_examples`; select `otsu`.



- [lab_05_manual:p9](extracted/page_images/lab_05_manual_p009.jpg)

## M05-HSV — HSV mask

Corrected implementation: `manual_examples.segmentation_examples`; select `hsv_mask`.



- [lab_05_manual:p9](extracted/page_images/lab_05_manual_p009.jpg)

## M05-CANNY — Canny edges

Corrected implementation: `manual_examples.segmentation_examples`; select `canny`.



- [lab_05_manual:p10](extracted/page_images/lab_05_manual_p010.jpg)

## M05-REGION — Region growing

Corrected implementation: `manual_examples.segmentation_examples`; select `region_grow`.



- [lab_05_manual:p10](extracted/page_images/lab_05_manual_p010.jpg)
- [lab_05_manual:p11](extracted/page_images/lab_05_manual_p011.jpg)
- [lab_05_manual:p12](extracted/page_images/lab_05_manual_p012.jpg)

## M05-WATER — Watershed markers

Corrected implementation: `manual_examples.segmentation_examples`; select `watershed`.



- [lab_05_manual:p13](extracted/page_images/lab_05_manual_p013.jpg)

## M05-KMEANS — K-means colors

Corrected implementation: `manual_examples.segmentation_examples`; select `kmeans`.



- [lab_05_manual:p15](extracted/page_images/lab_05_manual_p015.jpg)

## M06-SOBEL — Sobel direction and magnitude

Corrected implementation: `manual_examples.edge_feature_examples`; select `sobel_x;sobel_y;magnitude`.



- [lab_manual_06:p7](extracted/page_images/lab_manual_06_p007.jpg)
- [lab_manual_06:p8](extracted/page_images/lab_manual_06_p008.jpg)
- [lab_manual_06:p9](extracted/page_images/lab_manual_06_p009.jpg)
- [lab_manual_06:p10](extracted/page_images/lab_manual_06_p010.jpg)

## M06-CANNY — Gaussian and Canny

Corrected implementation: `manual_examples.edge_feature_examples`; select `canny`.



- [lab_manual_06:p12](extracted/page_images/lab_manual_06_p012.jpg)
- [lab_manual_06:p13](extracted/page_images/lab_manual_06_p013.jpg)

## M06-LOG — Laplacian after Gaussian

Corrected implementation: `manual_examples.edge_feature_examples`; select `laplacian_signed;laplacian_absolute_display;log_zero_crossings`.



- [lab_manual_06:p15](extracted/page_images/lab_manual_06_p015.jpg)
- [lab_manual_06:p16](extracted/page_images/lab_manual_06_p016.jpg)

## M06-SIFT — SIFT keypoint drawing

Corrected implementation: `manual_examples.edge_feature_examples`; select `sift_keypoints_image`.



- [lab_manual_06:p22](extracted/page_images/lab_manual_06_p022.jpg)
- [lab_manual_06:p23](extracted/page_images/lab_manual_06_p023.jpg)
