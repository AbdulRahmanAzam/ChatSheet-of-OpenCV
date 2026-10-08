# Worked task solutions

All page references below are **physical PDF pages**, starting at1. The original documents, full-page scans and raw OCR remain under sources/ and extracted/. These explanations and Python implementations are newly authored, corrected teaching solutions. Synthetic execution is not validation on the missing original datasets.

Run all demonstrations: `python solutions/run_synthetic.py`. For your own images, import a task function from its module; the function returns arrays/results. Use `cv_core.grid` to save image dictionaries. No internet call is used.

## L01-T01 — GroceryManager class

Source: lab_01_tasks:p1.

Implementation: `lab01.GroceryManager` in [solutions/lab01.py](solutions/lab01.py).

1. Store item quantity and unit price in a dictionary.
2. Validate input; add quantities when an item is repeated.
3. Remove by name and raise KeyError if absent.
4. Return a copy of the list and sum quantity times price.

**Assumptions and interpretation:** Supporting Python exercise; repeated-item behavior is an explicit implementation choice.

## L01-T02 — Nested student records and highest average

Source: lab_01_tasks:p1.

Implementation: `lab01.task02_students` in [solutions/lab01.py](solutions/lab01.py).

1. Compute a mean for each nonempty grade collection.
2. Find the highest mean and return all tied student names.
3. Filter records by a case-insensitive major.

**Assumptions and interpretation:** Empty grades excluded; ties preserved. Supporting Python exercise.

## L01-T03 — Load Sukuna image and format display

Source: lab_01_tasks:p2.

Implementation: `lab01.task03_display` in [solutions/lab01.py](solutions/lab01.py).

1. Decode Sukuna.jpeg; fail clearly on missing input.
2. Convert BGR to RGB.
3. Plot at 8x6 inches with axes hidden and dark-red 16-point title with pad 15.

**Assumptions and interpretation:** User must supply Sukuna.jpeg; synthetic demo substitutes a labeled generated image only for testing.

## L01-T04 — Five concentric circles and bounding box

Source: lab_01_tasks:p2.

Implementation: `lab01.task04_circles` in [solutions/lab01.py](solutions/lab01.py).

1. Create a black 800x800 color canvas.
2. Draw five filled alternating circles from largest to smallest.
3. Draw the outer bounding box and report center/bounds.

**Assumptions and interpretation:** Drawing center (400,400); geometric pixel-center midpoint is (399.5,399.5). Chosen radii 300,240,180,120,60.

## L01-T05 — 25x25 blur and exact center ROI

Source: lab_01_tasks:p2, lab_01_tasks:p3.

Implementation: `lab01.task05_blur_roi` in [solutions/lab01.py](solutions/lab01.py).

1. Check image is at least 300x300.
2. Blur the whole original with a 25x25 Gaussian.
3. Extract the same center 300x300 coordinates from both.
4. Plot side by side with titles and axes hidden.

**Assumptions and interpretation:** High-resolution course image missing; compare actual texture retention once supplied.

## L01-T06 — Transparent blue bottom banner

Source: lab_01_tasks:p3.

Implementation: `lab01.task06_overlay` in [solutions/lab01.py](solutions/lab01.py).

1. Copy the image and fill the bottom 20 percent with blue.
2. Blend copy/original with opacity and its complement.
3. Add readable text inside the banner.

**Assumptions and interpretation:** Opacity 0.4 and text are configurable. BGR blue=(255,0,0).

## L01-T07 — Thresholding and 45-degree scaled rotation

Source: lab_01_tasks:p3.

Implementation: `lab01.task07_threshold_rotate` in [solutions/lab01.py](solutions/lab01.py).

1. Convert to grayscale.
2. Compare global 127 with local Gaussian threshold.
3. Build getRotationMatrix2D at 45 degrees and scale 0.8.
4. Expand output bounds to preserve transformed content.

**Assumptions and interpretation:** Scale 0.8 alone does not prevent cropping on a same-size square canvas; expansion corrects that assumption.

## L01-T08 — 500x500 mask-based composition

Source: lab_01_tasks:p3.

Implementation: `lab01.task08_composite` in [solutions/lab01.py](solutions/lab01.py).

1. Resize both images to 500x500.
2. Draw a circular binary mask.
3. Select foreground with the mask and background with its inverse.
4. Combine disjoint selections using bitwise_or.

**Assumptions and interpretation:** Circle radius 180 chosen; distortion from fixed square resize follows the task.

## L01-T09 — RGB statistics and optional Pandas describe

Source: lab_01_tasks:p3, lab_01_tasks:p4.

Implementation: `lab01.task09_rgb_statistics` in [solutions/lab01.py](solutions/lab01.py).

1. Convert BGR to RGB and flatten to rows of pixels.
2. Calculate count, mean, sample standard deviation, min, quartiles and max.
3. The optional task09_optional_pandas follows the original DataFrame.describe request.

**Assumptions and interpretation:** Core uses NumPy to respect requested dependency scope; optional Pandas path is preserved but not exercised.

## L02-T01 — X-ray enhancement and pseudocolor

Source: lab_02_tasks:p4, lab_02_tasks:p5.

Implementation: `lab02.task01_xray` in [solutions/lab02.py](solutions/lab02.py).

1. Convert to grayscale and equalize histogram.
2. Apply JET colormap.
3. Apply explicitly simulated channel gains.
4. Threshold high intensities.
5. Compare log and gamma 0.6 transforms.

**Assumptions and interpretation:** 8-bit educational visualization; no lesion diagnosis or physical density calibration. Original images absent.

## L02-T02 — CT/MRI registered-slice fusion

Source: lab_02_tasks:p5, lab_02_tasks:p6.

Implementation: `lab02.task02_fusion` in [solutions/lab02.py](solutions/lab02.py).

1. Require registered corresponding slices of the same shape.
2. Equalize each grayscale image.
3. Apply distinct colormaps.
4. Blend CT 0.7 and MRI 0.3.
5. Show log and gamma versions.

**Assumptions and interpretation:** registered=True asserts external registration has been performed; this function cannot verify anatomy automatically.

## L02-T03 — Echo video visualization

Source: lab_02_tasks:p6, lab_02_tasks:p7.

Implementation: `lab02.task03_echo_video` in [solutions/lab02.py](solutions/lab02.py).

1. Open local video and validate decoder.
2. Process each frame with grayscale equalization, intensity pseudocolor, gains, log and gamma.
3. Concatenate original and processed views.
4. Save a silent video and release resources.

**Assumptions and interpretation:** Colors visualize intensity; not Doppler blood flow. Dataset links preserved in source pages; no dataset downloaded.

## L03-T01 — Manual matrix fingerprint enlargement

Source: lab_03_tasks:p3.

Implementation: `lab03.task01_scale` in [solutions/lab03.py](solutions/lab03.py).

1. Construct a diagonal 2D scale matrix and embed in 3x3 homogeneous form.
2. Translate source center to origin and destination center into place.
3. Warp to three-times width and height.

**Assumptions and interpretation:** Task says enlarge by 300 percent: default interprets as to 300 percent (factor 3); use 4 for a literal 300 percent increase.

## L03-T02 — Undo satellite rotation with sin/cos

Source: lab_03_tasks:p3.

Implementation: `lab03.task02_undo_rotation` in [solutions/lab03.py](solutions/lab03.py).

1. Build a 2x2 inverse rotation from sine/cosine.
2. Center the rotation using homogeneous translations.
3. Transform corners, compute bounds, and warp to an expanded canvas.

**Assumptions and interpretation:** Default corrects a 45 degree clockwise image rotation. Direction must match actual input.

## L03-T03 — Barcode horizontal de-shearing

Source: lab_03_tasks:p3.

Implementation: `lab03.task03_deshear` in [solutions/lab03.py](solutions/lab03.py).

1. Represent observed horizontal shear as xprime=x+k*y.
2. Use inverse coefficient minus k.
3. Warp to expanded output bounds.

**Assumptions and interpretation:** Shear coefficient is not supplied by task; default 0.3 is illustrative and configurable.

## L03-T04 — Map translation recovery

Source: lab_03_tasks:p3.

Implementation: `lab03.task04_translate` in [solutions/lab03.py](solutions/lab03.py).

1. Undo 150 pixels left and 80 pixels up using translation(+150,+80).
2. Create an output canvas that accommodates the positive shift.
3. Warp with the 3x3 matrix.

**Assumptions and interpretation:** Translation cannot restore image content that was already cropped away in the input.

## L03-T05 — Rigid transformation

Source: lab_03_tasks:p2.

Implementation: `lab03.task05_rigid` in [solutions/lab03.py](solutions/lab03.py).

1. Build rotation matrix.
2. Compose translation after rotation.
3. Apply once to the image and retain matrix for point checks.

**Assumptions and interpretation:** Parameters are examples because the exercise provides no numeric correspondences. Canvas may need adjustment.

## L03-T06 — Similarity transformation

Source: lab_03_tasks:p2.

Implementation: `lab03.task06_similarity` in [solutions/lab03.py](solutions/lab03.py).

1. Compose uniform scale, rotation, then translation.
2. Apply a single warp.
3. Verify all distances scale by the same positive factor.

**Assumptions and interpretation:** Default values illustrative. Similarity preserves angles but not absolute lengths.

## L03-T07 — Six-unknown affine solve

Source: lab_03_tasks:p2.

Implementation: `lab03.task07_affine` in [solutions/lab03.py](solutions/lab03.py).

1. Build six linear equations from three point pairs.
2. Reject collinear source landmarks.
3. Solve the six coefficients with np.linalg.solve.
4. Warp into the destination frame.

**Assumptions and interpretation:** Requires real corresponding landmarks; synthetic landmarks only validate the solver.

## L03-T08 — Top-down square perspective rectification

Source: lab_03_tasks:p2.

Implementation: `lab03.task08_rectify` in [solutions/lab03.py](solutions/lab03.py).

1. Provide four ordered convex corners.
2. Associate them with square corners.
3. Estimate getPerspectiveTransform and warpPerspective.

**Assumptions and interpretation:** Corner order TL,TR,BR,BL; target 400x400 default may distort a nonsquare physical object.

## L03-T09 — Panorama from field correspondences

Source: lab_03_tasks:p1.

Implementation: `lab03.task09_panorama` in [solutions/lab03.py](solutions/lab03.py).

1. Provide at least four paired landmarks.
2. Estimate right-to-left homography and reject failure.
3. Compute full canvas bounds.
4. Warp validity masks and average overlapping valid content.

**Assumptions and interpretation:** Planar scene or approximately pure camera rotation assumed; parallax and exposure seams remain limitations.

## L03-T10 — Painting insertion using transform composition

Source: lab_03_tasks:p1.

Implementation: `lab03.task10_painting` in [solutions/lab03.py](solutions/lab03.py).

1. Build linear scaling and rigid rotation/translation matrices.
2. Transform painting corners through those stages.
3. Estimate projective mapping from those intermediate corners to the wall quadrilateral.
4. Compose all matrices and warp original once.
5. Insert using a warped all-ones mask.

**Assumptions and interpretation:** Black painting pixels remain valid. Specifying the final four corners determines the final mapping; intermediate operations illustrate composition.

## L04-T01 — Material texture analysis with HOG and LBP

Source: lab_04_manual:p15.

Implementation: `lab04.task01_features` in [solutions/lab04.py](solutions/lab04.py).

1. Compute circular uniform LBP histogram for local microtexture.
2. Compute normalized block HOG for directional structure.
3. Compare optional local energy and contrast.
4. For an actual label, compare with labeled exemplars via classify_material.

**Assumptions and interpretation:** LBP: useful microtexture, weak under noise/scale change. HOG: useful grain/weave direction, weak under rotation and texture ambiguity. No wood/metal/fabric accuracy claimed without a labeled dataset.

## L05-T01 — Global versus adaptive document segmentation

Source: lab_05_tasks:p1, lab_05_tasks:p2.

Implementation: `lab05.task01_document` in [solutions/lab05.py](solutions/lab05.py).

1. Apply global inverse thresholds 80,127,180.
2. Apply adaptive Gaussian inverse threshold.
3. Compare masks and illumination failures; with known truth use task10_compare.

**Assumptions and interpretation:** Threshold polarity is dark text as foreground. Which method is best depends on the supplied document.

## L05-T02 — Adaptive parameter sweep

Source: lab_05_tasks:p2, lab_05_tasks:p3.

Implementation: `lab05.task02_adaptive_sweep` in [solutions/lab05.py](solutions/lab05.py).

1. Try mean and Gaussian local thresholds.
2. Cross block sizes 11,31,51 with C 2,7,12.
3. Display all 18 combinations and inspect broken/merged strokes.

**Assumptions and interpretation:** BINARY output: increasing C generally makes more white. No universal optimal block size.

## L05-T03 — Otsu coin segmentation and histogram changes

Source: lab_05_tasks:p3.

Implementation: `lab05.task03_otsu` in [solutions/lab05.py](solutions/lab05.py).

1. Convert to grayscale and record histogram.
2. Run Otsu and retain its numeric threshold.
3. Repeat after 5x5 blur and a documented contrast change.
4. Compare three masks and histograms.

**Assumptions and interpretation:** Contrast transform alpha 1.2 beta 10 is an experiment; clips highlights. Histograms are returned and plotted by run_synthetic.py.

## L05-T04 — HSV yellow-car isolation

Source: lab_05_tasks:p4.

Implementation: `lab05.task04_yellow` in [solutions/lab05.py](solutions/lab05.py).

1. Convert BGR to HSV.
2. Create a narrow yellow mask and broader yellow mask.
3. Extract pixels with each mask.
4. Inspect background leakage versus missed dim yellow areas.

**Assumptions and interpretation:** Fixed ranges are starting examples, not calibrated for missing car images. Hue is 0..179 for uint8.

## L05-T05 — Canny high-threshold comparison

Source: lab_05_tasks:p5.

Implementation: `lab05.task05_canny` in [solutions/lab05.py](solutions/lab05.py).

1. Convert and Gaussian smooth.
2. Hold low threshold 30 constant and try high 60,120,200.
3. Compare connected weak edges and suppression.

**Assumptions and interpretation:** Final output cannot identify original strong/weak classes; that would require access to intermediate gradients/NMS.

## L05-T06 — Two seeds and three region-growth tolerances

Source: lab_05_tasks:p6.

Implementation: `lab05.task06_region_grow` in [solutions/lab05.py](solutions/lab05.py).

1. Provide two seed points.
2. Grow with fixed-seed tolerances 5,15,30.
3. Compare six masks for leakage and incomplete coverage.

**Assumptions and interpretation:** Course MRI and seeds absent. This is intensity segmentation, not a clinical lesion classifier.

## L05-T07 — Complete marker-based watershed pipeline

Source: lab_05_tasks:p7.

Implementation: `lab05.task07_watershed` in [solutions/lab05.py](solutions/lab05.py).

1. Convert to grayscale and choose foreground polarity.
2. Otsu threshold; open the mask.
3. Dilate to background support.
4. Distance transform and threshold for foreground.
5. Subtract to find unknown.
6. Label seeds; add 1 so known background=1; unknown=0.
7. Run watershed on BGR uint8 with int32 markers.
8. Mark boundaries red and show all stages.

**Assumptions and interpretation:** Red is BGR(0,0,255). Empty/poor markers can leave objects unseparated.

## L05-T08 — Distance-threshold effects on watershed

Source: lab_05_tasks:p8.

Implementation: `lab05.task08_distance_sweep` in [solutions/lab05.py](solutions/lab05.py).

1. Repeat full watershed for fractions 0.2,0.4,0.6,0.8 of maximum distance.
2. Record foreground marker count and final labeled-region count.
3. Compare overlays and explain merged/lost seeds.

**Assumptions and interpretation:** A higher fraction is not always better; small objects can lose all seeds. Counts depend on image.

## L05-T09 — K=2,4,6 image clustering

Source: lab_05_tasks:p9.

Implementation: `lab05.task09_kmeans` in [solutions/lab05.py](solutions/lab05.py).

1. Reshape BGR pixels to float32 rows.
2. Run K-means for 2,4,6 with recorded seed and stopping criteria.
3. Reconstruct center-colored images.
4. Compare compactness, color detail and false merges.

**Assumptions and interpretation:** Spatial position is not included in this baseline; disconnected areas of the same color share a label.

## L05-T10 — Compare three segmentation methods

Source: lab_05_tasks:p10, lab_05_tasks:p11.

Implementation: `lab05.task10_compare` in [solutions/lab05.py](solutions/lab05.py).

1. Choose a dark-text document as the common task.
2. Compare fixed 127, Otsu and adaptive Gaussian masks.
3. Compute IoU only when ground truth is supplied.
4. Explain global failures under illumination changes and adaptive block/C tradeoffs.

**Assumptions and interpretation:** Includes measured synthetic results in validation.json; original-data comparison remains pending. No method wins universally.

## L06-T01 — Hough-supported screen monitoring

Source: lab_06_tasks:p1.

Implementation: `lab06.task01_screens` in [solutions/lab06.py](solutions/lab06.py).

1. Read image, grayscale, smooth and Canny.
2. Detect Hough line segments and draw them on a black canvas.
3. Check support around expected screen rectangles.
4. Measure interior brightness.
5. Display supported boundaries and unconfirmed locations.

**Assumptions and interpretation:** Requires calibrated front-facing ROIs. Outputs visible_dark/visible_bright/unconfirmed; absence may be missing or occluded, brightness is not proven power state.

## L06-T02 — SIFT asset inventory

Source: lab_06_tasks:p1.

Implementation: `lab06.task02_inventory` in [solutions/lab06.py](solutions/lab06.py).

1. Provide named reference pictures for assets.
2. Compute SIFT descriptors and match with BF L2 ratio test.
3. Validate planar localization by RANSAC.
4. Return evidence for each reference and explicit failure reasons.

**Assumptions and interpretation:** Identical-looking assets cannot be uniquely identified by appearance alone. Multiple instances and untextured screens require extensions.

## L06-T03 — Wavelet sensor anomalies

Source: lab_06_tasks:p1, lab_06_tasks:p2.

Implementation: `lab06.task03_wavelet_anomalies` in [solutions/lab06.py](solutions/lab06.py).

1. Decompose clean baseline with Haar DWT.
2. Estimate noise from detail coefficients and soft-threshold for a denoised output.
3. Separately reconstruct approximation-only signals and compute detail residuals.
4. Calibrate a cutoff using clean-baseline residual MAD.
5. Apply the same decomposition to the test signal and return deviations.

**Assumptions and interpretation:** Signal-processing appendix preserved from course. Baseline assumed representative; broad/slow anomalies can be missed. Soft-threshold denoising residuals are not used as anomaly scores because they can cap retained impulses.

## L06-T04 — Reference object recognition in images/video

Source: lab_06_tasks:p2.

Implementation: `lab06.task04_recognize` in [solutions/lab06.py](solutions/lab06.py).

1. Extract SIFT reference and scene features.
2. Apply ratio test, RANSAC and quadrilateral plausibility checks.
3. Draw localized boundary when evidence passes.
4. Use task04_video for a local video loop.

**Assumptions and interpretation:** Requires textured planar reference; returns failure if descriptors/matches are insufficient.

## L06-T05 — Automatic panorama from overlapping images

Source: lab_06_tasks:p2.

Implementation: `lab06.task05_panorama` in [solutions/lab06.py](solutions/lab06.py).

1. Use SIFT to align each next image to the current panorama.
2. Reject weak homographies.
3. Expand canvas and blend valid overlap.
4. Record inlier/match counts per step.

**Assumptions and interpretation:** Sequential baseline can drift; no bundle adjustment, exposure compensation or parallax handling.

## L06-T06 — Hough lane-line detection

Source: lab_06_tasks:p2, lab_06_tasks:p3.

Implementation: `lab06.task06_lanes` in [solutions/lab06.py](solutions/lab06.py).

1. Smooth and detect edges.
2. Restrict to a road-shaped trapezoid.
3. Detect segments and group by slope/side.
4. Fit x as a function of y and draw left/right boundaries.

**Assumptions and interpretation:** Fixed-camera straight-lane teaching example, not a vehicle-control system. ROI/slope assumptions fail on curves/hills.

## L06-T07 — Coin counting with Hough circles

Source: lab_06_tasks:p3.

Implementation: `lab06.task07_coins` in [solutions/lab06.py](solutions/lab06.py).

1. Grayscale and median blur.
2. Run HoughCircles with plausible radius bounds and center separation.
3. Safely handle no detections.
4. Draw centers/circumferences and count candidates.

**Assumptions and interpretation:** Perspective, touching coins and reflections change performance; tune on labeled actual images.

## L06-T08 — Restricted-zone visual-change monitor

Source: lab_06_tasks:p3.

Implementation: `lab06.ZoneMonitor` in [solutions/lab06.py](solutions/lab06.py).

1. Provide empty background and zone polygon.
2. Compute absolute frame/background difference.
3. Threshold and open the mask; restrict to zone.
4. Filter small components.
5. Require consecutive frames and emit one local event per active episode.

**Assumptions and interpretation:** Explicit policy is any new foreground in the zone; visual change does not prove a person/object is unauthorized. Requires static camera. No external alarm is sent.
