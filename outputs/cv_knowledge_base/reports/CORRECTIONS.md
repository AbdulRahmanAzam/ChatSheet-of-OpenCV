# Corrections and ambiguities

The original scans are unchanged. These are curated corrections, not edits to course source text.

## C01 (lab_01_manual:p10)

Coordinate diagram can confuse row/column with x/y.

**Corrected interpretation:** Use image[y,x], with x column, y row; OpenCV points use (x,y).

## C02 (lab_01_manual:p14)

Prose describes cv.imshow/waitKey but displayed code uses Matplotlib.

**Corrected interpretation:** The reconstructed example saves a Matplotlib figure; GUI calls are a separate alternative.

## C03 (lab_01_manual:p17)

Comment describes5x5 blur while code uses31x31.

**Corrected interpretation:** Record actual example as31x31; Task05 independently requests25x25.

## C04 (lab_01_manual:p19)

Text scale in code/prose differs.

**Corrected interpretation:** Reconstructed example uses a documented visible scale1.5; preserve original scan for exact original constant.

## C05 (lab_01_manual:p21)

Threshold maxval200 is described as pure white.

**Corrected interpretation:** uint8 white is255;200 is a lighter gray although auto-scaled plots may show it white.

## C06 (lab_01_manual:p23)

cv.add is presented as image blending.

**Corrected interpretation:** cv.add is saturating addition; cv.addWeighted with complementary weights implements alpha blending.

## C07 (lab_01_tasks:p3)

Scale0.8 at45degrees assumed to prevent cropping.

**Corrected interpretation:** For a square,0.8*sqrt(2)>1; expand canvas or use a smaller scale if preserving all content.

## C08 (lab_03_tasks:p3)

Enlarge by300percent is ambiguous.

**Corrected interpretation:** Default interprets final size300percent (3x); strict increase by300percent means4x. Both are documented.

## C09 (lab_03_tasks:p1, lab_03_tasks:p2, lab_03_tasks:p3)

Physical PDF page order reverses exercise groups.

**Corrected interpretation:** Canonical task order1..10 maps to physical pages3,2,1. All references use physical PDF pages.

## C10 (lab_03_manual:p17)

Homogeneous notation motivates sweeping performance claims about vector addition/GPU cost.

**Corrected interpretation:** The reliable benefit here is unified representation and composition; speed depends on implementation and hardware.

## C11 (lab_04_manual:p11, lab_04_manual:p13)

Signed atan2 angles are histogrammed over positive-only ranges.

**Corrected interpretation:** Wrap angles modulo360 or180 before binning; otherwise negative angles are discarded.

## C12 (lab_04_manual:p11)

HED creates a Canny map but histograms all gradient orientations.

**Corrected interpretation:** Restrict to selected edge pixels for an edge-direction histogram, and document weighting.

## C13 (lab_04_manual:p13)

Histogram normalization implies scale invariance.

**Corrected interpretation:** Normalization reduces dependence on sample count/overall weighting; it does not generally make texture spatially scale invariant.

## C14 (lab_04_manual:p13, lab_04_manual:p14)

High sum-of-squared intensity energy is equated to complex texture.

**Corrected interpretation:** Uniform white has high intensity energy and zero contrast; distinguish brightness energy from variation.

## C15 (lab_04_manual:p14, lab_04_manual:p15)

uint8 squared intensities overflow, output may saturate, and contrast code sums pixels.

**Corrected interpretation:** Convert to float before squaring; local standard deviation=sqrt(max(E[I²]-E[I]²,0)).

## C16 (lab_04_manual:p17)

A getGaussianKernel column is used as if a full2D blur.

**Corrected interpretation:** Use outer product v@v.T or sepFilter2D with both directions.

## C17 (lab_04_manual:p18, lab_04_manual:p19)

filter2D same-size output is labeled full/valid; reflected borders called zero padding.

**Corrected interpretation:** Explicitly pad for full, crop for valid, use BORDER_CONSTANT for zero padding; same describes shape, not border rule.

## C18 (lab_04_manual:p19)

filter2D is passed strides=(2,2), an unsupported argument.

**Corrected interpretation:** Filter then subsample explicit valid output, or use another operation that actually exposes stride.

## C19 (lab_04_manual:p16, lab_04_manual:p18)

filter2D called convolution without the correlation distinction.

**Corrected interpretation:** Flip an asymmetric kernel in both axes for true convolution.

## C20 (lab_04_manual:p21)

Sobel-Feldman described as a separate directional variation.

**Corrected interpretation:** Sobel-Feldman is the Sobel operator name; derivative directions are selected explicitly.

## C21 (lab_manual_06:p11)

Non-maximum suppression described only as eliminating weak pixels.

**Corrected interpretation:** NMS keeps local maxima along gradient direction; hysteresis thresholds then select connected edges.

## C22 (lab_05_manual:p11)

Region-growth coordinate names differ from OpenCV point convention.

**Corrected interpretation:** Bundled API consistently accepts(x,y), indexes[y,x], checks seed, and marks queued coordinates visited.

## C23 (lab_05_manual:p13)

Boundary [255,0,0] called red in a BGR array; Matplotlib display is unconverted.

**Corrected interpretation:** BGR red=(0,0,255), followed by BGR2RGB when plotting. Display conversion and array convention must agree.

## C24 (lab_05_manual:p15)

K-means BGR array sent directly to Matplotlib.

**Corrected interpretation:** Convert BGR to RGB for plotting; retain BGR for subsequent OpenCV operations.

## C25 (lab_manual_06:p15, lab_manual_06:p16)

Absolute LoG image can be mistaken for zero-crossing edges.

**Corrected interpretation:** Preserve signed response and explicitly detect sign changes for zero-crossing localization.

## C26 (lab_manual_06:p17)

Slope/intercept line description omits vertical-line limitation.

**Corrected interpretation:** OpenCV polar Hough uses rho/theta, covering vertical lines without infinite slope.

## C27 (lab_manual_06:p21)

Illustration small text says120-dimensional while grid is4x4x8.

**Corrected interpretation:** Standard SIFT descriptor is128-dimensional.

## C28 (lab_manual_06:p23)

descriptors.shape accessed unconditionally.

**Corrected interpretation:** No keypoints may return descriptors=None; guard before shape/matching.

## C29 (lab_06_tasks:p1)

Line boundaries/brightness alone treated as enough for screen missing/on-off status.

**Corrected interpretation:** Require expected layout and report visual evidence; missing vs occluded and dark vs powered-off are distinct.

## C30 (lab_02_tasks:p6)

Intensity colorization can be read as blood-flow information.

**Corrected interpretation:** JET of grayscale carries only intensity; it is not measured Doppler velocity.

## C31 (lab_02_tasks:p5)

Fusion can be attempted without explicit registration.

**Corrected interpretation:** Require already registered corresponding slices; resizing alone is insufficient.

## C32 (lab_manual_06:p1)

Lab Manual06 filename has a cover labeled Lab05.

**Corrected interpretation:** Keep the supplied filename identity lab_manual_06; map its wavelet/edge/Hough/SIFT content to the supplied Lab06 tasks, and preserve the cover discrepancy.