# Computer vision knowledge handbook

Scope: supplied labs plus related OpenCV/NumPy/Matplotlib fundamentals. This is a curated offline corpus, not a universal computer-vision model. Source references distinguish course grounding from additional documentation. Code fragments assume imports and input context stated in records/topics.jsonl.

## Pixels, resolution, coordinates and channels

An image is a sampled array. A grayscale image has shape (height,width); BGR color has (height,width,3). Pixel depth means the numeric type/precision, while geometric depth is distance in a 3D scene. Resolution controls spatial sampling, not semantic detail recovered from nothing.

**Use when:** Before indexing, cropping, resizing, or interpreting ddepth.

**Pitfalls:** NumPy indexes row then column; OpenCV drawing points are (x,y). uint8 arithmetic can wrap. Pixel coordinate origin is top-left.

Setup (earlier steps this example assumes):

```python
x, y = 10, 20
```

```python
height, width = image.shape[:2]
pixel = image[y, x]
blue, green, red = image[y, x]
```

Sources: lab_01_manual:p10.

## Read and save an image

Decode the file into an array and check for failure before using its shape or converting colors. Standard color reads are BGR. Grayscale reads request a single channel. Saving an array encodes it according to the output filename extension.

**Use when:** Start any image-processing exercise.

**Pitfalls:** A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Do not silently replace missing course input with synthetic data.

Setup (earlier steps this example assumes):

```python
cv.imwrite('input.png', image)   # stand-in for your own image file
```

```python
image = cv.imread('input.png', cv.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError('input.png')
if not cv.imwrite('output.png', image):
    raise OSError('Could not save image')
```

Sources: lab_01_manual:p14.

## Plot images correctly

OpenCV color arrays normally use BGR, while Matplotlib interprets a three-channel array as RGB. Convert only for plotting. For comparable grayscale panels set a fixed range. plt.subplots returns a figure and axes, which is convenient for labeled comparisons.

**Use when:** Inspect processing stages and export a reproducible figure.

**Pitfalls:** Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. cv.imshow needs a GUI-enabled OpenCV build; saved Matplotlib figures work headlessly.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
ax[1].imshow(g, cmap='gray', vmin=0, vmax=255)
for a in ax: a.axis('off')
fig.tight_layout()
fig.savefig('comparison.png')
plt.close(fig)
```

Sources: lab_01_manual:p14, lab_manual_06:p10.

## Color conversion and HSV ranges

Grayscale combines channels into luminance-like intensity for many geometric operations. HSV separates hue, saturation, and value for color masks. VGA describes a display resolution/standard, not a color space; the intended operation in the earlier example is BGR to RGB.

**Use when:** Use grayscale for thresholding/gradients; HSV for color-based selection.

**Pitfalls:** For OpenCV uint8 HSV, hue is 0..179 and saturation/value are 0..255. Floating HSV uses different scaling. A red hue range often wraps and needs two masks.

```python
g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
```

Sources: lab_01_manual:p15, lab_05_manual:p9.

## NumPy arrays and safe image arithmetic

NumPy stores images and performs vectorized arithmetic. Convert to floating point before squaring, subtracting, or averaging when intermediate values exceed uint8. reshape changes layout without changing element order; ravel may return a view. Clip and round before converting results back to uint8.

**Use when:** Gradient magnitude, pixel clustering, feature vectors, masks, and normalization.

**Pitfalls:** np.uint8([-1]) or a cast of out-of-range floats does not provide meaningful saturation. Squaring uint8 before conversion has already lost information.

Setup (earlier steps this example assumes):

```python
dx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
dy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
```

```python
f = g.astype(np.float32)
squared = f*f
magnitude = np.hypot(dx, dy)
result = np.clip(np.rint(f), 0, 255).astype(np.uint8)
pixels = image.reshape(-1, 3).astype(np.float32)
```

Sources: lab_04_manual:p14, lab_manual_06:p10.

## Output depth and signed responses

ddepth chooses the output numeric depth. -1 keeps the input depth, which is suitable for some smoothing but clips negative derivative responses when the input is unsigned. CV_32F or CV_64F preserves negative gradients and large intermediate results.

**Use when:** Sobel, Scharr, Laplacian, filter2D, texture statistics.

**Pitfalls:** Output depth does not mean neural-network depth or physical distance. An absolute display image loses the sign needed for orientation and zero crossings.

```python
dx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
display = cv.convertScaleAbs(dx)
```

Sources: lab_manual_06:p8, lab_04_manual:p17.

## Cropping and region of interest

A region of interest is a rectangular or masked subset of an image. Slicing uses [y_start:y_end,x_start:x_end], with exclusive end indices. A center crop of size s starts at ((width-s)//2,(height-s)//2). R10 is unresolved terminology in the user request; it is not automatically treated as an API.

**Use when:** Inspect fine detail or limit processing to a relevant area.

**Pitfalls:** A slice may be a view: modifying it can modify the original image. Clamping coordinates silently can change the requested crop size.

Setup (earlier steps this example assumes):

```python
image = cv.resize(image, (640, 480))   # crop needs an image of at least 300x300
```

```python
h, w = image.shape[:2]
s = 300
if min(h,w) < s: raise ValueError("image too small")
x, y = (w-s)//2, (h-s)//2
roi = image[y:y+s, x:x+s].copy()
```

Sources: lab_01_manual:p18, lab_01_tasks:p2.

## Resize and interpolation

Resizing resamples the image to a different pixel grid. INTER_AREA is often useful for shrinking, linear/cubic for natural-image enlargement, and nearest neighbor for label masks. A larger output does not recover lost fine structure.

**Use when:** Standardize input dimensions or show a scaled image.

**Pitfalls:** dsize is (width,height). Resizing two medical images to the same size does not register their anatomy.

Setup (earlier steps this example assumes):

```python
_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

```python
small = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
mask2 = cv.resize(mask, (500, 500), interpolation=cv.INTER_NEAREST)
```

Sources: lab_01_manual:p16, lab_03_manual:p6.

## Draw circles, rectangles, lines and text

Draw into a writable array using pixel positions and BGR color tuples. A negative thickness fills supported shapes. Text needs an origin on the baseline, a font, scale, color, and thickness.

**Use when:** Annotate detections, create masks, or draw a synthetic test image.

**Pitfalls:** Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center.

```python
canvas = np.zeros((800,800,3), np.uint8)
cv.circle(canvas, (400,400), 100, (0,255,0), 2)
cv.putText(canvas, "screen", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
```

Sources: lab_01_tasks:p2, lab_01_manual:p19.

## Weighted blending versus image addition

cv.add adds intensities with saturation for uint8. Alpha blending uses weighted inputs and an optional offset. To add a transparent banner, copy the image, draw the banner on the copy, then blend it with the original.

**Use when:** Overlays, registered image fusion, gradual transitions.

**Pitfalls:** Addition is not a 50/50 blend. Inputs need matching size/type. Register images before interpreting anatomical or geometric correspondence.

```python
overlay = image.copy()
overlay[int(image.shape[0]*0.8):] = (255,0,0)
out = cv.addWeighted(overlay, 0.4, image, 0.6, 0)
```

Sources: lab_01_manual:p23, lab_02_tasks:p5.

## Binary masks and bitwise composition

A mask selects locations. A uint8 single-channel mask commonly uses 0 for excluded and 255 for included. inRange produces such a mask from channel bounds. Complementary masks can choose foreground pixels from one image and background pixels from another.

**Use when:** Color extraction, object cutouts, ROI selection, compositing.

**Pitfalls:** Mask dimensions must match the image. AND/OR on arbitrary intensities are bit operations, not probabilistic blending.

Setup (earlier steps this example assumes):

```python
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
```

```python
mask = cv.inRange(hsv, np.array([20,80,80]), np.array([35,255,255]))
selected = cv.bitwise_and(image, image, mask=mask)
inverse = cv.bitwise_not(mask)
```

Sources: lab_01_tasks:p3, lab_05_manual:p9.

## Image histograms and bar plots

A histogram counts samples in bins; it discards spatial location. Intensity histograms summarize brightness, color histograms summarize channel distributions, and edge-direction histograms summarize orientation. Histogram bin edges are not image edges.

**Use when:** Inspect contrast, choose thresholds, compare distributions, or form descriptors.

**Pitfalls:** Specify range and normalization consistently. Different images can have the same histogram. Count histograms and density histograms have different units.

```python
hist, edges = np.histogram(g, bins=256, range=(0,256))
import matplotlib.pyplot as plt
plt.bar(edges[:-1], hist, width=1)
plt.xlabel("Intensity"); plt.ylabel("Pixel count")
```

Sources: lab_04_manual:p9, lab_05_tasks:p3.

## Histogram equalization and CLAHE

Histogram equalization remaps grayscale intensities using their cumulative distribution to spread occupied values. CLAHE adapts locally with a contrast limit. Neither creates missing image information, and noise may become more visible.

**Use when:** Improve visible contrast when the input occupies a narrow intensity range.

**Pitfalls:** equalizeHist expects uint8 single-channel input. Independently equalizing BGR channels alters colors; work on a luminance channel when color fidelity matters.

```python
equalized = cv.equalizeHist(g)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
local = clahe.apply(g)
```

Sources: lab_01_manual:p24, lab_02_tasks:p4.

## Log and gamma intensity transforms

Using output=255*(input/255)^gamma, gamma below one brightens midtones and gamma above one darkens them. A logarithmic transform expands dark values and compresses highlights. Both are point operations without spatial understanding.

**Use when:** Visualize low-intensity detail or compress a display range.

**Pitfalls:** Some libraries define a reciprocal gamma parameter; state the convention. Convert before arithmetic. Pseudocolor and display transforms are not measurements of disease or flow.

```python
f = g.astype(np.float32)/255
out = np.clip(255*f**0.6, 0, 255).astype(np.uint8)
log_image = (255*np.log1p(g.astype(np.float32))/np.log(256)).astype(np.uint8)
```

Sources: lab_02_tasks:p4, lab_02_tasks:p6.

## Pseudocolor and intensity visualization

A colormap assigns display colors to scalar intensities. It can make differences easier to see but adds no measured spectral, anatomical, or flow information. Channel gains applied afterward are a visualization choice.

**Use when:** The Lab 02 grayscale imaging demonstrations.

**Pitfalls:** JET colors in an echocardiogram are not Doppler measurements. Keep the original scalar image available and avoid interpreting chosen colors as clinical labels.

```python
heatmap = cv.applyColorMap(g, cv.COLORMAP_JET)
```

Sources: lab_02_tasks:p4, lab_02_tasks:p6.

## Kernels, correlation and convolution

A kernel assigns weights to a local neighborhood. Correlation multiplies the neighborhood by the kernel directly; convolution flips the kernel along both axes first. Symmetric blur kernels give the same result either way. OpenCV filter2D performs correlation.

**Use when:** Smoothing, sharpening, derivatives, and custom local responses.

**Pitfalls:** A negative/signed kernel with uint8 output clips information. Normalize averaging kernels. The manual incorrectly treats all filter2D examples as different convolution sizes.

```python
k = np.ones((3,3),np.float32)/9
smooth = cv.filter2D(g, cv.CV_32F, k)
true_convolution = cv.filter2D(g, cv.CV_32F, k[::-1,::-1])
```

Sources: lab_04_manual:p16, lab_04_manual:p18.

## Full, same, valid, stride and border handling

For odd kernel size k and stride one, valid output size is n-k+1, same is n, and full is n+k-1. Border rules supply values outside the image: constant, replicate, reflect including the edge, or reflect101 excluding the edge. Output shape and border rule are separate choices.

**Use when:** Reproduce textbook convolution dimensions or reason about edge artifacts.

**Pitfalls:** filter2D has no strides keyword and always keeps spatial size. BORDER_CONSTANT alone does not make valid convolution. Unsupported border modes vary by function; do not assume WRAP is accepted everywhere.

Setup (earlier steps this example assumes):

```python
k = np.ones((3, 3), np.float32) / 9
```

```python
from cv_core import convolution
valid = convolution(g, k, mode='valid', stride=2)
padded = cv.copyMakeBorder(g, 2,2,2,2, cv.BORDER_REFLECT_101)
```

Sources: lab_04_manual:p18, lab_04_manual:p19.

## Box blur

A normalized box kernel averages all pixels in a rectangular neighborhood equally. It reduces small variations but softens boundaries. The supplied manual calls this Box Blur; botch filter appears to be a spelling/transcription ambiguity, not a separate verified operator.

**Use when:** Fast simple smoothing or local means.

**Pitfalls:** An unnormalized box filter returns a sum, which can overflow or saturate. It is not a local standard-deviation calculation.

```python
blurred = cv.blur(image, (5,5))
mean = cv.boxFilter(g.astype(np.float32), -1, (5,5), normalize=True)
```

Sources: lab_04_manual:p16.

## Gaussian blur and Gaussian kernels

Gaussian smoothing weights nearby pixels most strongly and suppresses high-frequency variation. sigma controls the spread; the odd kernel size truncates its support. A 2D Gaussian is separable into horizontal and vertical 1D filtering.

**Use when:** Denoise before gradients, Canny, or scale-space analysis.

**Pitfalls:** getGaussianKernel returns one column, so passing it alone to filter2D smooths only one direction. Larger sigma removes more detail.

```python
smooth = cv.GaussianBlur(image, (5,5), 1.0)
v = cv.getGaussianKernel(5, 1.0)
k2d = v @ v.T
equivalent = cv.sepFilter2D(image, -1, v, v)
```

Sources: lab_01_manual:p17, lab_04_manual:p17.

## Median filtering

A median filter selects the median value in a neighborhood rather than averaging it. It is useful for isolated high/low impulse noise and can preserve sharp transitions better than linear averaging.

**Use when:** Salt-and-pepper noise or preprocessing for some circle-detection tasks.

**Pitfalls:** Kernel size must be odd and greater than one. Large windows remove small features. Supported input depths depend on kernel size.

```python
smooth = cv.medianBlur(image, 5)
```

Sources: lab_06_tasks:p3.

## Bilateral filtering

Bilateral filtering combines spatial proximity and similarity in intensity/color. Nearby pixels across a large color discontinuity contribute less, allowing some smoothing within regions while retaining boundaries.

**Use when:** Denoising when preserving significant intensity boundaries matters.

**Pitfalls:** Usually slower than a small Gaussian filter. Large range sigma makes it behave more like spatial smoothing; it can flatten subtle texture that a later descriptor needs.

```python
smooth = cv.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
```

Sources: supplement:filtering.

## Sobel derivatives and gradient magnitude

Sobel estimates first spatial derivatives using a small derivative kernel with smoothing in the perpendicular direction. dx responds to left-right intensity change, often revealing vertical boundaries; dy reveals horizontal ones. Magnitude combines them; atan2 gives signed orientation.

**Use when:** Edges, orientation histograms, HOG, and geometric boundary cues.

**Pitfalls:** Sobel-Feldman is another name for Sobel, not a separate diagonal detector. A derivative response is not yet a thin binary edge map.

```python
dx = cv.Sobel(g, cv.CV_32F, 1,0,ksize=3)
dy = cv.Sobel(g, cv.CV_32F, 0,1,ksize=3)
magnitude = np.hypot(dx,dy)
angle = np.degrees(np.arctan2(dy,dx))
```

Sources: lab_04_manual:p20, lab_04_manual:p21, lab_manual_06:p6.

## Scharr and Prewitt operators

Scharr uses derivative weights designed to improve rotational accuracy of a 3x3 approximation. Prewitt uses simpler uniform smoothing weights. These yield signed derivatives, which can be combined into gradient magnitude.

**Use when:** Compare compact gradient operators; Scharr is useful when a small kernel is required.

**Pitfalls:** SCAR in the request corresponds to the Scharr topic present in the manual. Derivative scales differ; the same threshold is not automatically comparable across operators.

```python
sx = cv.Scharr(g, cv.CV_32F, 1,0)
sy = cv.Scharr(g, cv.CV_32F, 0,1)
prewitt_x = cv.filter2D(g, cv.CV_32F, np.float32([[-1,0,1],[-1,0,1],[-1,0,1]]))
```

Sources: lab_04_manual:p17, lab_04_manual:p20.

## Laplacian and Laplacian of Gaussian

The Laplacian sums second spatial derivatives. Smoothing first reduces noise sensitivity, producing a discrete Laplacian-of-Gaussian style pipeline. Edge localization by zero crossings uses changes in the sign of the response, often with a minimum response-range threshold.

**Use when:** Detect changes in gradient or explore second-derivative edges.

**Pitfalls:** The absolute response visualization is not a zero-crossing edge map. Keep signed float response for zero-crossing tests.

```python
blur = cv.GaussianBlur(g,(5,5),1.4)
response = cv.Laplacian(blur,cv.CV_32F)
visualization = cv.convertScaleAbs(response)
```

Sources: lab_manual_06:p14, lab_manual_06:p15, lab_manual_06:p16.

## Absolute scaling for visualization

convertScaleAbs applies scaling and offset, takes absolute values, and saturates into uint8. It is convenient for displaying signed gradients, but destroys polarity and clips values outside the display range.

**Use when:** Show a Sobel or Laplacian response after keeping its signed original.

**Pitfalls:** Do not compute orientation or zero crossings from the absolute uint8 image.

Setup (earlier steps this example assumes):

```python
dx = cv.Sobel(g, cv.CV_16S, 1, 0, ksize=3)
```

```python
view = cv.convertScaleAbs(dx, alpha=1, beta=0)
```

Sources: lab_manual_06:p15.

## Canny edge detection

A Canny pipeline smooths noise, computes gradients, thins local maxima along the gradient direction, then retains strong edges and weak edges connected to them. Low and high thresholds govern hysteresis. The result is a thin binary edge map, not filled object regions.

**Use when:** Prepare line detection or inspect boundaries.

**Pitfalls:** Use explicit blur to control smoothing. Non-maximum suppression is not simply a weak-pixel threshold. Final Canny output alone does not label which surviving pixels began strong or weak.

```python
blur = cv.GaussianBlur(g,(5,5),1.4)
edges = cv.Canny(blur,50,150,L2gradient=True)
```

Sources: lab_04_manual:p22, lab_manual_06:p11, lab_manual_06:p13.

## Global thresholding and threshold types

A global threshold applies one cutoff everywhere. Binary output chooses zero or maxval; inverse binary swaps those assignments. Truncation caps high values, and to-zero keeps only values on a selected side. Global binary thresholding works best with separable foreground/background intensities.

**Use when:** Simple segmentation under reasonably uniform illumination.

**Pitfalls:** maxval=200 means intensity200, not255. A plotting program may rescale it to white. Uneven light can defeat any single cutoff.

```python
value, mask = cv.threshold(g,127,255,cv.THRESH_BINARY)
inverted = cv.threshold(g,127,255,cv.THRESH_BINARY_INV)[1]
```

Sources: lab_01_manual:p21, lab_05_manual:p8.

## Adaptive mean and Gaussian thresholding

Adaptive thresholding computes a local mean or Gaussian-weighted mean and subtracts C. Each pixel is compared with that local threshold. Small neighborhoods follow fine variation but can fragment strokes; very large ones approach broad illumination averaging.

**Use when:** Documents with shadows or spatially varying brightness.

**Pitfalls:** blockSize must be odd and greater than1, with uint8 single-channel input. Increasing C lowers the threshold: more white pixels for BINARY and fewer for BINARY_INV.

```python
mask = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,7)
```

Sources: lab_05_manual:p8, lab_05_manual:p9, lab_05_tasks:p2.

## Otsu automatic threshold selection

Otsu chooses a global threshold by optimizing the separation of two intensity classes. It uses the image histogram and returns the selected numeric threshold as well as the binary mask. It can work well when foreground/background form distinct groups.

**Use when:** Automatically segment high-contrast grayscale objects under fairly uniform illumination.

**Pitfalls:** Otsu does not fix spatially varying illumination or guarantee semantic objects. Blur/contrast transforms change the histogram and can change the selected threshold.

```python
t, mask = cv.threshold(g,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
```

Sources: lab_05_manual:p9, lab_05_tasks:p3.

## Morphological erosion, dilation, opening and closing

On a binary foreground mask, dilation expands foreground and erosion shrinks it according to a structuring element. Opening is erosion then dilation, removing small foreground details. Closing is dilation then erosion, filling small gaps. Neighborhood shape and iteration count control the spatial scale.

**Use when:** Clean threshold masks, join broken regions, and prepare watershed markers.

**Pitfalls:** Foreground polarity matters. Large kernels can remove real small objects or merge neighbors. Repeated operations change geometry.

Setup (earlier steps this example assumes):

```python
_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

```python
k = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
opened = cv.morphologyEx(mask,cv.MORPH_OPEN,k)
closed = cv.morphologyEx(mask,cv.MORPH_CLOSE,k)
```

Sources: lab_05_manual:p13, lab_05_tasks:p7.

## Morphological gradient, top-hat and black-hat

The morphological gradient is dilation minus erosion, highlighting a band around boundaries. Top-hat is the original minus its opening, emphasizing small bright structures. Black-hat is closing minus the original, emphasizing small dark structures.

**Use when:** Boundary visualization and simple uneven-background correction.

**Pitfalls:** Choose an element larger than the structures being separated from the background. These are morphological operations, not arbitrary linear convolution kernels.

```python
k = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
bright = cv.morphologyEx(g,cv.MORPH_TOPHAT,k)
dark = cv.morphologyEx(g,cv.MORPH_BLACKHAT,k)
```

Sources: supplement:morphology.

## Seeded region growing

Start at a seed and visit connected neighbors that satisfy a similarity rule. The supplied solution compares each candidate with the fixed seed intensity, avoiding accidental drift of an evolving mean. Four-connectivity uses axial neighbors; eight-connectivity also uses diagonals.

**Use when:** Select one connected fairly uniform region when a seed is known.

**Pitfalls:** Seed order is (x,y) here; the manual uses array-style coordinates. Convert intensity values to int before subtraction. Wrong seed or tolerance leaks or undersegments. Fixed seed and evolving-mean algorithms differ.

```python
from cv_core import region_grow
mask = region_grow(image, seed=(100,80), tolerance=15, connectivity=4)
```

Sources: lab_05_manual:p10, lab_05_manual:p11.

## Distance transforms and foreground seeds

A distance transform gives each nonzero pixel its distance to the nearest zero pixel. Thresholding high distances finds interior regions of sufficiently thick foreground objects, often useful as watershed seeds.

**Use when:** Create markers inside blobs and separate touching objects.

**Pitfalls:** The input must have the correct foreground polarity and a surrounding zero background. A high fraction can erase small-object seeds; a low fraction can merge touching-object seeds.

Setup (earlier steps this example assumes):

```python
_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

```python
dist = cv.distanceTransform(mask,cv.DIST_L2,5)
sure_fg = np.uint8(dist > 0.5*dist.max())*255
```

Sources: lab_05_manual:p13, lab_05_tasks:p8.

## Connected components and integer labels

Connected-component labeling gives each connected nonzero region an integer label; zero denotes background. The returned count includes background. Stats can report bounding boxes and pixel areas.

**Use when:** Count separated mask blobs, form watershed seeds, or filter small regions.

**Pitfalls:** Touching objects can share one component. A component count is not necessarily a real-object count.

Setup (earlier steps this example assumes):

```python
_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

```python
count, labels, stats, centers = cv.connectedComponentsWithStats(mask, connectivity=8)
object_count = count-1
```

Sources: lab_05_manual:p13.

## Marker-controlled watershed segmentation

Provide labeled confident interiors and an unknown area. Watershed propagates labels through image variation, marking separating boundaries. The teaching pipeline uses thresholding, opening, dilated background support, distance-based foreground, connected components, and unknown pixels.

**Use when:** Separate touching objects when useful seeds can be created.

**Pitfalls:** cv.watershed takes a uint8 3-channel image and int32 markers. Unknown starts at0, boundaries become-1. In this pipeline background is1. It mutates markers. Poor seeds cause poor segmentation.

```python
from cv_core import watershed_stages
stages = watershed_stages(image, distance_fraction=0.5, foreground="bright")
labels = stages["markers"]
```

Sources: lab_05_manual:p12, lab_05_manual:p13, lab_05_tasks:p7.

## K-means color clustering

K-means alternates assigning pixels to the nearest center and updating centers, minimizing within-cluster squared distances. Reshape color pixels to N-by-3 float32, cluster, replace each pixel by its center, and reshape back.

**Use when:** Color quantization and an unsupervised segmentation baseline.

**Pitfalls:** Color clusters are not guaranteed semantic objects or connected regions. K controls model size, and initialization can affect results. Evaluate K choices instead of assuming larger is always better.

```python
from cv_core import segment_kmeans
segmented, labels, compactness = segment_kmeans(image, k=4, seed=42)
```

Sources: lab_05_manual:p14, lab_05_manual:p15, lab_05_tasks:p9.

## Segmentation versus edges, detection and classification

Segmentation assigns labels to pixels; detection locates instances, often with boxes; classification assigns a category to an image or region. Semantic segmentation gives class labels, while instance segmentation distinguishes individual objects. An edge map only identifies local boundaries.

**Use when:** Choose the actual output required before choosing an algorithm.

**Pitfalls:** A color cluster or threshold region does not inherently know whether it is a screen, lesion, or coin. Labels need assumptions, reference data, or a trained classifier.

```python
mask = cv.threshold(g,127,255,cv.THRESH_BINARY)[1]
```

Sources: lab_01_manual:p5, lab_05_manual:p3, lab_05_manual:p5.

## Contours and geometric boundaries

Contours trace boundaries in a binary mask. Their area, perimeter, polygon approximation, and bounding boxes support shape analysis. Polygon approximation can reduce a quadrilateral boundary to four vertices, but the semantic identity still needs context.

**Use when:** Find object outlines and candidate screen/document boundaries.

**Pitfalls:** RETR_EXTERNAL ignores nested holes. A bounding rectangle is not a perspective quadrilateral; contours and bounding-box pixel areas differ.

Setup (earlier steps this example assumes):

```python
_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

```python
contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    polygon = cv.approxPolyDP(contour,0.02*cv.arcLength(contour,True),True)
```

Sources: supplement:contours, lab_06_tasks:p1.

## Homogeneous coordinates and matrix composition

Use [x,y,1] to represent 2D points so translation and linear transforms compose as 3x3 matrices. With column vectors, the rightmost matrix acts first. Divide by the third coordinate after a projective transform.

**Use when:** Combine scale, rotation, translation, shear, and perspective in a single resampling.

**Pitfalls:** Image y increases downward. A standard Cartesian positive-angle matrix therefore appears clockwise on an image. Avoid repeated image warps when matrices can be composed.

Setup (earlier steps this example assumes):

```python
x, y = 10.0, 20.0
H = np.array([[1.0, 0.1, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
```

```python
p = np.array([x,y,1.0])
q = H @ p
q = q[:2]/q[2]
```

Sources: lab_03_manual:p15, lab_03_manual:p17.

## Linear, rigid, similarity and affine transformations

A linear 2D map uses a 2x2 matrix and fixes the origin. A rigid map adds rotation and translation while preserving distances. Similarity adds uniform scaling. An affine map additionally allows nonuniform scaling and shear and preserves parallel lines. Three noncollinear point pairs determine a 2D affine map.

**Use when:** Geometric correction when perspective effects are absent or negligible.

**Pitfalls:** Affine does not preserve every length or angle. Three collinear points are degenerate. Same-size output can clip transformed content.

```python
src = np.float32([[0,0],[100,0],[0,100]])
dst = np.float32([[20,10],[120,20],[10,110]])
M = cv.getAffineTransform(src,dst)
warped = cv.warpAffine(image,M,(image.shape[1],image.shape[0]))
```

Sources: lab_03_manual:p5, lab_03_manual:p11, lab_03_manual:p13, lab_03_tasks:p2.

## Perspective transformations and homographies

A 3x3 homography maps between views of a plane or images related by pure camera rotation. It preserves straight lines but can change angles, length ratios, and parallelism. Four nondegenerate point pairs determine it up to scale.

**Use when:** Rectify documents/paintings, map a planar reference into a scene, or stitch suitable views.

**Pitfalls:** Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography. Degenerate geometry and outliers require rejection.

```python
src = np.float32([[10,20],[300,15],[310,220],[20,230]])
dst = np.float32([[0,0],[399,0],[399,399],[0,399]])
H = cv.getPerspectiveTransform(src,dst)
warped = cv.warpPerspective(image,H,(400,400))
```

Sources: lab_03_manual:p18, lab_03_tasks:p1, lab_03_tasks:p2.

## Image registration and resampling

Registration estimates a geometric relationship between corresponding content. Select a model appropriate to motion and scene structure, estimate it from reliable correspondences, validate alignment, then resample. Medical slices also require physical orientation and correspondence, beyond equal array dimensions.

**Use when:** Image fusion, mosaics, change detection, and reference comparisons.

**Pitfalls:** Different modalities may not share feature appearance. A plausible-looking overlay is not proof of accurate registration. The fusion exercise expects already registered arrays.

Setup (earlier steps this example assumes):

```python
src_points = np.float32([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40], [20, 60], [80, 10], [30, 25]])
dst_points = src_points * 1.2 + np.float32([15, 8])
```

```python
H, inliers = cv.findHomography(src_points,dst_points,cv.RANSAC,3.0)
```

Sources: lab_02_tasks:p5, lab_03_tasks:p2.

## Histogram of Oriented Gradients

HOG aggregates gradient orientations within cells, weights contributions by gradient strength, and normalizes groups of neighboring cells in blocks. The resulting vector describes local edge structure while reducing sensitivity to overall contrast.

**Use when:** Handcrafted shape/texture features for a separately trained classifier.

**Pitfalls:** A descriptor alone does not classify an object. Window, cell, block, stride and bin choices must match training. OpenCV and scikit-image configurations are not automatically interchangeable.

```python
from cv_core import hog_features
features = hog_features(image)
```

Sources: lab_04_manual:p6, lab_04_manual:p7.

## Local Binary Patterns

Compare neighboring intensities with a center pixel to form binary patterns. Uniform circular patterns have at most two bit transitions; with eight neighbors they can be summarized by ten bins including one nonuniform bin. The histogram describes local microtexture.

**Use when:** Material texture baselines and local pattern comparisons.

**Pitfalls:** Noise changes threshold comparisons. Rotation properties depend on variant. The bundled replacement uses circular sampling with a cropped border and does not promise bit-for-bit equivalence with scikit-image.

```python
from cv_core import uniform_lbp
labels, histogram = uniform_lbp(image)
```

Sources: lab_04_manual:p7, lab_04_manual:p8, lab_04_manual:p9.

## Edge-direction and intensity-gradient histograms

An edge-direction histogram accumulates gradient angles only at selected edge pixels. A broader gradient histogram can use all nonzero gradients and magnitude weights. Wrap signed atan2 output into 0..360 for directed angles or 0..180 for unsigned orientation before binning.

**Use when:** Summarize dominant boundary directions and texture orientations.

**Pitfalls:** The manual drops negative angles by using a positive range without wrapping and computes HED without applying its Canny mask. HED also names a different learned edge detector in other literature; this course means histogram of edge directions.

```python
from cv_core import direction_histogram
hed, bins = direction_histogram(image,8,edge_only=True,signed=True,weighted=False)
hig, bins = direction_histogram(image,9,weighted=True)
```

Sources: lab_04_manual:p9, lab_04_manual:p11, lab_04_manual:p13.

## Texture energy, contrast and classification

Local intensity energy can mean the sum of squared intensities; local contrast can be measured by standard deviation. Compute means and mean squares in float, then variance=max(E[I²]-E[I]²,0). A uniformly bright patch has high intensity energy but zero contrast, so energy alone does not imply complex texture.

**Use when:** Compare material patterns, with labeled training data and controlled sampling.

**Pitfalls:** The manual contrast code is a local sum, not the stated standard deviation, and its uint8 square can overflow. Other fields use different energy definitions such as squared filter responses: name the definition.

```python
from cv_core import texture_stats
energy, contrast = texture_stats(image,7)
```

Sources: lab_04_manual:p13, lab_04_manual:p14, lab_04_manual:p15.

## Gabor texture filters

A Gabor filter combines a sinusoidal pattern with a Gaussian envelope to respond to a chosen orientation and spatial frequency. A bank of orientations/scales can represent textures with repeated directional structure.

**Use when:** Supplement LBP/HOG for fabric weave or wood grain.

**Pitfalls:** Scale and orientation choices matter. Responses need normalization and a classifier for material labels. More filters increase descriptor size and compute time.

```python
k = cv.getGaborKernel((21,21),4,np.pi/4,10,0.5,0,ktype=cv.CV_32F)
response = cv.filter2D(g,cv.CV_32F,k)
```

Sources: supplement:filtering.

## Hough line and segment detection

Line Hough voting accumulates evidence from edge points in parameter space. A polar representation rho=x*cos(theta)+y*sin(theta) handles vertical lines. HoughLines returns polar lines; HoughLinesP returns finite segment endpoints.

**Use when:** Straight boundaries, lane candidates, and calibrated screen-side evidence.

**Pitfalls:** A Hough line is not automatically a screen or lane. Texture can vote strongly, repeated edges cause duplicates, and lines can be absent. Handle None.

```python
edges = cv.Canny(g,50,150)
lines = cv.HoughLinesP(edges,1,np.pi/180,40,minLineLength=40,maxLineGap=12)
if lines is not None:
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        cv.line(image,(x1,y1),(x2,y2),(0,255,0),2)
```

Sources: lab_manual_06:p17, lab_06_tasks:p1, lab_06_tasks:p2.

## Screen boundaries, brightness and missing-screen hypotheses

Use camera calibration/reference layout to define expected screen regions. Check Hough-supported sides and examine interior brightness. If a reference region lacks evidence, label it unconfirmed: the screen may be missing, occluded, displaced, or obscured by lighting.

**Use when:** A constrained fixed-camera monitoring prototype.

**Pitfalls:** Dark content is not proof the power is off, and a bright reflection is not proof it is on. This solution uses calibrated front-facing rectangles; automatic perspective-screen discovery and real-data validation remain extensions.

```python
from lab06 import task01_screens
result = task01_screens(image, [(30,60,180,150)])
```

Sources: lab_06_tasks:p1.

## Hough circle detection

The gradient Hough circle method finds circle centers/radii from grayscale structure. dp controls accumulator resolution, minDist separates centers, param1 relates to the internal edge threshold, and param2 controls detection acceptance for the standard gradient method.

**Use when:** Count circular candidates with a plausible radius range.

**Pitfalls:** Parameter meanings differ for other circle methods. Perspective turns circles into ellipses. Filter duplicate detections and handle None. Avoid unsigned integer arithmetic when drawing/checking coordinates.

```python
g = cv.medianBlur(g,5)
circles = cv.HoughCircles(g,cv.HOUGH_GRADIENT,1.2,30,param1=120,param2=25,minRadius=10,maxRadius=80)
```

Sources: lab_manual_06:p18, lab_06_tasks:p3.

## SIFT keypoints and descriptors

SIFT detects stable local structures across a scale space, refines their positions, assigns orientations, and describes local gradient distributions. A standard descriptor has 128 components from 4x4 spatial cells with 8 orientation bins.

**Use when:** Match textured planar references across moderate scale/view changes.

**Pitfalls:** Flat or repetitive surfaces give weak/ambiguous matches. descriptors can be None. SIFT invariance is approximate and does not identify identical-looking physical assets by itself.

```python
sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(g,None)
drawn = cv.drawKeypoints(image,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

Sources: lab_manual_06:p18, lab_manual_06:p20, lab_manual_06:p23.

## Brute-force matching and ratio tests

A brute-force matcher compares descriptors. For float SIFT descriptors use L2 distance. Retrieve the two nearest candidates and retain a best match only when it is sufficiently better than the second; geometry then rejects additional outliers.

**Use when:** Reference recognition, inventory baselines and panorama alignment.

**Pitfalls:** The Python constructor is cv.BFMatcher, not cv.BruteForceMatcher. Guard empty descriptors and short pairs. Binary ORB descriptors normally use Hamming distance, with Hamming2 for ORB WTA_K=3/4.

Setup (earlier steps this example assumes):

```python
sift = cv.SIFT_create()
rotated = cv.warpAffine(g, cv.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), 8, 1.0), (g.shape[1], g.shape[0]))
_, desc1 = sift.detectAndCompute(g, None)
_, desc2 = sift.detectAndCompute(rotated, None)
```

```python
bf = cv.BFMatcher(cv.NORM_L2)
pairs = bf.knnMatch(desc1,desc2,k=2)
good = [p[0] for p in pairs if len(p)==2 and p[0].distance < 0.75*p[1].distance]
```

Sources: lab_manual_06:p21, lab_06_tasks:p1.

## RANSAC and geometric verification

RANSAC repeatedly estimates geometry from small samples and scores consistent correspondences. A final model supported by enough inliers is more reliable than descriptor similarity alone. Check projected shape and canvas size as well as match counts.

**Use when:** Reject false feature matches before localization/stitching.

**Pitfalls:** Four matches suffice algebraically but not necessarily for reliability. Collinear/repetitive points and parallax can fool a model. Inspect inlier distribution and reprojection error on real data.

Setup (earlier steps this example assumes):

```python
src_points = np.float32([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40], [20, 60], [80, 10], [30, 25]])
dst_points = src_points * 1.2 + np.float32([15, 8])
```

```python
H, inliers = cv.findHomography(src_points,dst_points,cv.RANSAC,4.0)
if H is None: raise ValueError("No reliable homography")
```

Sources: lab_manual_06:p21, lab_manual_06:p22.

## ORB as a compact alternative

ORB combines corner detection and binary descriptors to provide an efficient local feature baseline. Binary descriptors are compact and matched using bit differences, unlike SIFT float vectors.

**Use when:** When CPU speed or descriptor storage matters more than SIFT robustness.

**Pitfalls:** Use the correct distance norm; ORB and SIFT descriptors cannot be mixed. Accuracy depends on image content and transformation.

```python
orb = cv.ORB_create(nfeatures=1000)
kp, desc = orb.detectAndCompute(g,None)
bf = cv.BFMatcher(cv.NORM_HAMMING)
```

Sources: supplement:matching.

## Panorama stitching and valid-pixel masks

Match overlapping views, verify a homography, compute the combined transformed bounds, translate them into a positive canvas, and warp both image and validity mask. Blend only where both images contribute valid pixels.

**Use when:** The manual-point and SIFT-based panorama exercises.

**Pitfalls:** Black pixels can be valid image content, so intensity>0 is not a validity mask. Sequential stitching accumulates drift; parallax, exposure differences and moving objects need more advanced handling.

Setup (earlier steps this example assumes):

```python
rng = np.random.default_rng(1)
scene = cv.GaussianBlur(rng.integers(0, 256, (240, 480, 3), dtype=np.uint8), (5, 5), 0)
image1, image2 = scene[:, :300].copy(), scene[:, 180:].copy()   # overlapping views
```

```python
from lab06 import task05_panorama
panorama, evidence = task05_panorama([image1,image2])
```

Sources: lab_03_tasks:p1, lab_06_tasks:p2.

## Haar wavelets, multiscale analysis and denoising

Wavelets separate coarse approximations from localized detail at multiple scales. A Haar pair uses (a+b)/sqrt(2) and (a-b)/sqrt(2). DWT recursively decomposes the approximation; wavelet packets also decompose details. CWT samples a family of scales/shifts more densely.

**Use when:** Multiscale texture or the supplied one-dimensional sensor exercise.

**Pitfalls:** Thresholding removes some detail and may suppress anomalies themselves. Boundary padding and scale affect output. The sensor exercise is preserved as an appendix because it is not image-specific.

Setup (earlier steps this example assumes):

```python
rng = np.random.default_rng(0)
clean_baseline = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)
signal = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)
signal[100] += 3.0   # injected impulse
```

```python
from cv_core import haar_denoise
denoised, threshold = haar_denoise(signal,levels=3)
```

Sources: lab_manual_06:p3, lab_manual_06:p4, lab_06_tasks:p1.

## Baseline-calibrated anomaly scores

Compare a measured statistic with a baseline distribution. The teaching wavelet example thresholds the signal minus its approximation-only Haar reconstruction using robust median absolute deviation estimated from clean baseline data. Denoising is a separate output. It returns indices and scores rather than claiming a cause.

**Use when:** Explore transient sensor deviations or explicitly defined visual-change policies.

**Pitfalls:** A single threshold cannot detect every anomaly type. Slow drift may stay in the approximation. Evaluate false alarms and missed events on labeled sequences.

Setup (earlier steps this example assumes):

```python
rng = np.random.default_rng(0)
clean_baseline = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)
signal = np.sin(np.linspace(0, 8 * np.pi, 256)) + rng.normal(0, 0.05, 256)
signal[100] += 3.0   # injected impulse
```

```python
from lab06 import task03_wavelet_anomalies
result = task03_wavelet_anomalies(signal,clean_baseline)
```

Sources: lab_06_tasks:p1, lab_06_tasks:p2.

## Offline video processing and frame loops

Open a local video, check that the decoder opened, read until decoding ends, process each frame, and release the capture/writer even if an error occurs. Output frames must keep the writer size/type consistent.

**Use when:** Echo visualization, reference recognition, lanes, or zone monitoring.

**Pitfalls:** Codec support depends on the installed build. This writer creates silent processed video; it does not preserve audio. Camera movement invalidates a fixed background reference.

Setup (earlier steps this example assumes):

```python
h, w = image.shape[:2]
writer = cv.VideoWriter('input.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
for i in range(5):
    writer.write(np.roll(image, 4 * i, axis=1))
writer.release()
def frame_function(frame):
    return cv.Canny(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 50, 150)
```

```python
from cv_core import process_video
count = process_video("input.mp4", frame_function, "output.mp4")
```

Sources: lab_02_tasks:p6, lab_06_tasks:p2.

## Evaluate masks, matches and monitoring systems

Use labeled evaluation data that represents intended conditions. For binary masks, IoU is intersection divided by union. Detection/matching needs localization and false-positive checks. Monitoring also needs sequence-level false alarms, misses, and latency.

**Use when:** Choose among methods and tune parameters without guessing performance.

**Pitfalls:** Synthetic tests verify controlled behavior, not real-world accuracy. Do not select thresholds on the same data used to claim final performance.

Setup (earlier steps this example assumes):

```python
_, mask = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
predicted = mask
truth = np.roll(mask, 3, axis=1)
```

```python
a, b = predicted > 0, truth > 0
union = np.count_nonzero(a | b)
iou = np.count_nonzero(a & b)/union if union else 1.0
```

Sources: lab_05_tasks:p10, lab_05_tasks:p11.

## Camera calibration, distortion and physical scale

Camera calibration estimates focal parameters and lens distortion from multiple views of known geometry. Undistortion corrects curved projections of straight structures before geometric measurement. Physical size estimates also need distance, plane geometry, or a calibrated reference.

**Use when:** Improve screen/document geometry and measurements when lens distortion is significant.

**Pitfalls:** Calibration arrays must come from actual calibration, not invented constants. Pixel width alone is not a physical measurement.

Setup (earlier steps this example assumes):

```python
h, w = image.shape[:2]
camera_matrix = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
distortion_coefficients = np.array([-0.1, 0.01, 0, 0, 0])   # from cv.calibrateCamera in practice
```

```python
corrected = cv.undistort(image,camera_matrix,distortion_coefficients)
```

Sources: supplement:geometry.

## Related CV topics: conceptual coverage and expansion boundary

The manuals also mention learned recognition/segmentation, motion tracking, optical flow, restoration, compression, active contours and graph/clustering methods. Their source discussions are preserved, but the current executable set focuses on the 41 supplied exercises. Learned models require separately obtained weights and representative evaluation data. Optical flow, mean-shift/CamShift tracking, background subtraction, Haar cascades, GrabCut, stereo depth and cv.dnn inference now have dedicated supplemental topic records.

**Use when:** Choose a later expansion when handcrafted image evidence is insufficient.

**Pitfalls:** This corpus does not contain every algorithm or answer in computer vision. Do not present conceptual mentions as implemented methods. No deep-learning weights or general language model are included.

Sources: lab_01_manual:p5, lab_01_manual:p11, lab_01_manual:p12, lab_05_manual:p5, lab_05_manual:p16.

## Fourier transform and frequency-domain filtering

The 2D discrete Fourier transform represents an image as a sum of sinusoids. Low frequencies (near the centre after fftshift) carry smooth structure; high frequencies carry edges, texture and noise. Multiplying the spectrum by a mask and inverting gives frequency-domain filtering. By the convolution theorem, spatial convolution equals spectral multiplication.

**Use when:** Inspect periodic noise, design ideal/Gaussian low- or high-pass filters, or explain why blurring removes detail.

**Pitfalls:** Display log magnitude, not raw magnitude. Forgetting ifftshift before ifft2 shifts the output. An ideal (hard-edged) mask causes ringing; a Gaussian mask avoids it. Take the real part and clip before converting to uint8.

```python
f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
magnitude = 20 * np.log1p(np.abs(f))
rows, cols = g.shape
cy, cx = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.float32)
cv.circle(mask, (cx, cy), 30, 1, -1)          # ideal low-pass
low = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
high = np.real(np.fft.ifft2(np.fft.ifftshift(f * (1 - mask))))
low_u8 = np.clip(low, 0, 255).astype(np.uint8)
```

Sources: supplement:fourier.

## Image pyramids (Gaussian and Laplacian)

A Gaussian pyramid repeatedly blurs and halves an image. A Laplacian pyramid stores the detail lost at each level (level minus upsampled next level), allowing exact reconstruction. Pyramids support coarse-to-fine search, multi-scale detection and seamless blending.

**Use when:** Search for objects at several sizes, speed up processing on a coarse level, or blend two images without a visible seam.

**Pitfalls:** pyrUp does not undo pyrDown; detail is lost unless stored in the Laplacian level. Store Laplacian levels as signed types, since uint8 subtraction clips negatives. Pass dstsize for odd image sizes.

```python
levels = [g]
for _ in range(3):
    levels.append(cv.pyrDown(levels[-1]))
laplacian = []
for i in range(3):
    h, w = levels[i].shape[:2]
    up = cv.pyrUp(levels[i + 1], dstsize=(w, h))
    laplacian.append(cv.subtract(levels[i], up, dtype=cv.CV_16S))
# reconstruct the finest level from the next level plus stored detail
h, w = levels[0].shape[:2]
rebuilt = (cv.pyrUp(levels[1], dstsize=(w, h)).astype(np.int16) + laplacian[0]).astype(np.uint8)
```

Sources: supplement:pyramids.

## Template matching

matchTemplate slides a template over the image and scores every position. With TM_CCOEFF_NORMED the best match is the maximum (1.0 is perfect); with TM_SQDIFF/TM_SQDIFF_NORMED the best match is the minimum. The result map is (H-h+1, W-w+1).

**Use when:** Locate a known, same-scale, same-orientation pattern such as an icon, logo or fixed part.

**Pitfalls:** Not invariant to scale or rotation; use pyramids or features for those. For SQDIFF methods take min_loc, not max_loc. minMaxLoc returns (x, y) while np.where returns (rows, cols). Repeated detections near one location need non-maximum suppression.

```python
template = g[100:150, 20:100].copy()
h, w = template.shape
res = cv.matchTemplate(g, template, cv.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
marked = image.copy()
cv.rectangle(marked, top_left, bottom_right, (0, 0, 255), 2)
# all matches above a threshold
ys, xs = np.where(res >= 0.9)
```

Sources: supplement:template.

## Harris and Shi-Tomasi corner detection

Corners have strong intensity change in two directions. Harris computes a response R = det(M) - k*trace(M)^2 from the local gradient structure tensor M. Shi-Tomasi uses the smaller eigenvalue min(l1,l2) and goodFeaturesToTrack returns the strongest well-separated corners. cornerSubPix refines positions below pixel resolution.

**Use when:** Choose points to track, calibrate with checkerboards, or explain what makes a good feature.

**Pitfalls:** cornerHarris needs float32 input and returns a response map, not a point list; threshold relative to R.max(). Corners are not scale invariant. goodFeaturesToTrack may return None on flat images.

```python
gray32 = np.float32(g)
R = cv.cornerHarris(gray32, blockSize=2, ksize=3, k=0.04)
harris = image.copy()
harris[R > 0.01 * R.max()] = (0, 0, 255)
pts = cv.goodFeaturesToTrack(g, maxCorners=50, qualityLevel=0.01, minDistance=10)
if pts is not None:
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    pts = cv.cornerSubPix(g, pts, (5, 5), (-1, -1), criteria)
    for x, y in pts.reshape(-1, 2):
        cv.circle(harris, (int(x), int(y)), 4, (0, 255, 0), 1)
```

Sources: supplement:corners.

## FAST, AKAZE and BRISK feature detectors

FAST is a very quick corner detector without descriptors. AKAZE builds a nonlinear scale space and produces binary descriptors; BRISK produces binary descriptors with scale/rotation handling. Binary descriptors (ORB, AKAZE, BRISK) are matched with Hamming distance; float descriptors (SIFT) with L2.

**Use when:** Choose a detector by speed, invariance, licensing and descriptor type.

**Pitfalls:** FAST alone gives no descriptor, so it cannot be matched without a separate extractor. Using NORM_L2 on binary descriptors gives meaningless distances. Descriptors can be None when no keypoints are found.

```python
fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
kp_fast = fast.detect(g, None)
akaze = cv.AKAZE_create()
kp_a, des_a = akaze.detectAndCompute(g, None)
brisk = cv.BRISK_create()
kp_b, des_b = brisk.detectAndCompute(g, None)
vis = cv.drawKeypoints(image, kp_fast, None, color=(0, 255, 0))
matcher = cv.BFMatcher(cv.NORM_HAMMING)   # binary descriptors
```

Sources: supplement:features2d.

## FLANN matching and drawing matches

FLANN performs approximate nearest-neighbour search, faster than brute force on large descriptor sets. For float descriptors (SIFT) use a KD-tree index; for binary descriptors use LSH. Combine knnMatch(k=2) with Lowe ratio test, then visualize with drawMatches.

**Use when:** Match many descriptors quickly, for example in panoramas or object recognition.

**Pitfalls:** FLANN KD-tree requires float32 descriptors; binary descriptors need algorithm=6 (LSH). knnMatch can return fewer than 2 neighbours, so check the pair length. Results are approximate and can vary slightly.

```python
sift = cv.SIFT_create()
img2 = cv.warpAffine(g, cv.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), 10, 1.0), (g.shape[1], g.shape[0]))
kp1, d1 = sift.detectAndCompute(g, None)
kp2, d2 = sift.detectAndCompute(img2, None)
good = []
if d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    for pair in flann.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
vis = cv.drawMatches(g, kp1, img2, kp2, good[:30], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

Sources: supplement:matching.

## Moments, centroids and shape descriptors

Image moments summarise a contour: m00 is area, (m10/m00, m01/m00) is the centroid, and Hu moments are invariant to translation, scale and rotation. Shape measures such as aspect ratio, extent (area/bounding-box area), solidity (area/hull area) and circularity (4*pi*area/perimeter^2) help classify objects.

**Use when:** Measure, filter or classify segmented objects, for example counting coins or rejecting non-rectangular screen candidates.

**Pitfalls:** Guard m00 == 0 before dividing. fitEllipse needs at least 5 points. findContours finds white objects on black, so invert if needed. Hu moments span many orders of magnitude; compare their log values.

```python
_, bw = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
out = image.copy()
for c in contours:
    area = cv.contourArea(c)
    if area < 50:
        continue
    M = cv.moments(c)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    hull = cv.convexHull(c)
    solidity = area / max(cv.contourArea(hull), 1e-6)
    circularity = 4 * np.pi * area / max(cv.arcLength(c, True) ** 2, 1e-6)
    box = cv.boxPoints(cv.minAreaRect(c)).astype(np.int32)
    cv.drawContours(out, [box], -1, (0, 255, 0), 2)
    cv.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    hu = cv.HuMoments(M).ravel()
```

Sources: supplement:contours.

## Channels, flips, normalization, LUTs and borders

split/merge separate and rebuild channels. flip mirrors (0 vertical, 1 horizontal, -1 both). normalize rescales values, e.g. NORM_MINMAX to 0..255. LUT maps every uint8 value through a 256-entry table, a fast way to apply gamma or custom curves. copyMakeBorder pads explicitly: CONSTANT (fixed value), REPLICATE (edge pixel), REFLECT (abc|cba), REFLECT_101 (abc|ba, the default), WRAP (periodic).

**Use when:** Prepare inputs, visualize channel content, pad before filtering, or apply intensity curves efficiently.

**Pitfalls:** cv.split is slower than NumPy indexing for a single channel. LUT tables must be uint8 of length 256. hconcat requires equal heights and types. Filters use BORDER_REFLECT_101 by default, not zero padding.

```python
b, gch, r = cv.split(image)
swapped = cv.merge([r, gch, b])
mirror = cv.flip(image, 1)
rot90 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
stretched = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
gamma = 0.5
table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
brighter = cv.LUT(g, table)
padded = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0))
side_by_side = cv.hconcat([g, brighter])
```

Sources: supplement:border.

## Sharpening and unsharp masking

Sharpening adds back high-frequency detail. Unsharp masking subtracts a blurred copy: sharp = image + amount*(image - blur), implemented with addWeighted. A 3x3 kernel [[0,-1,0],[-1,5,-1],[0,-1,0]] is the image plus a negative Laplacian.

**Use when:** Enhance soft edges or text before display or detection.

**Pitfalls:** Sharpening amplifies noise; denoise first. Strong amounts create halos. Overshoot is clipped in uint8, so compute in float if exact values matter.

```python
blur = cv.GaussianBlur(image, (0, 0), 3)
unsharp = cv.addWeighted(image, 1.5, blur, -0.5, 0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
sharp = cv.filter2D(image, -1, kernel)
```

Sources: supplement:filtering.

## Noise models, denoising and PSNR

Gaussian noise adds normally distributed values; salt-and-pepper sets random pixels to 0 or 255; speckle multiplies. Median filtering suits impulse noise, Gaussian/bilateral suit Gaussian noise, and non-local means averages similar patches anywhere nearby. PSNR = 10*log10(255^2/MSE) measures fidelity to a clean reference (higher is better).

**Use when:** Create test inputs, choose a denoiser, and quantify the result.

**Pitfalls:** Add noise in float and clip, otherwise uint8 wraps around. PSNR needs a clean reference and does not always match perceived quality. Non-local means is slow on large images.

```python
rng = np.random.default_rng(0)
noisy = np.clip(g.astype(np.float32) + rng.normal(0, 15, g.shape), 0, 255).astype(np.uint8)
sp = g.copy()
m = rng.random(g.shape)
sp[m < 0.02] = 0
sp[m > 0.98] = 255
nlm = cv.fastNlMeansDenoising(noisy, None, h=15, templateWindowSize=7, searchWindowSize=21)
med = cv.medianBlur(sp, 3)
print('PSNR noisy', cv.PSNR(g, noisy), 'denoised', cv.PSNR(g, nlm))
```

Sources: supplement:denoising.

## GrabCut foreground extraction

GrabCut models foreground and background colors with Gaussian mixture models and solves a graph cut. Initialize with a rectangle around the object (or a labelled mask), iterate, then keep pixels labelled definite or probable foreground.

**Use when:** Interactive or box-guided object cut-out.

**Pitfalls:** Input must be 8-bit 3-channel. The rectangle must be inside the image and contain the object. Similar foreground/background colors confuse the model; correct with mask hints and GC_INIT_WITH_MASK.

```python
mask = np.zeros(image.shape[:2], np.uint8)
bgd = np.zeros((1, 65), np.float64)
fgd = np.zeros((1, 65), np.float64)
h, w = image.shape[:2]
rect = (w // 8, h // 8, w * 3 // 4, h * 3 // 4)
cv.grabCut(image, mask, rect, bgd, fgd, 3, cv.GC_INIT_WITH_RECT)
fg = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
cutout = cv.bitwise_and(image, image, mask=fg)
```

Sources: supplement:grabcut.

## Image inpainting

Inpainting fills masked pixels from their surroundings. INPAINT_TELEA uses fast marching; INPAINT_NS uses a Navier-Stokes based method. The mask is 8-bit with non-zero pixels marking damage.

**Use when:** Remove scratches, text overlays or small unwanted objects.

**Pitfalls:** Only plausible for thin or small regions; large holes become smeared. The mask must cover the whole defect, so dilate it slightly if needed.

```python
damaged = image.copy()
defect = np.zeros(image.shape[:2], np.uint8)
cv.line(defect, (10, 10), (image.shape[1] - 10, image.shape[0] - 10), 255, 3)
damaged[defect > 0] = (255, 255, 255)
restored = cv.inpaint(damaged, defect, 3, cv.INPAINT_TELEA)
```

Sources: supplement:inpaint.

## Haar cascade object detection

A Haar cascade evaluates rectangular intensity-difference features in a sequence of increasingly strict stages, rejecting most windows early. detectMultiScale scans across positions and scales and returns (x, y, w, h) boxes. OpenCV ships trained frontal-face, eye and other cascades.

**Use when:** Fast classical face or object detection on CPU when a trained cascade exists.

**Pitfalls:** Check classifier.empty(); a wrong path fails silently otherwise. Detection is sensitive to pose and lighting; equalizeHist can help. minNeighbors trades false positives for misses. Modern DNN detectors are usually more accurate.

```python
path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
face = cv.CascadeClassifier(path)
if face.empty():
    raise FileNotFoundError(path)
boxes = face.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
out = image.copy()
for (x, y, w, h) in boxes:
    cv.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
```

Sources: supplement:cascades.

## Background subtraction and frame differencing

Frame differencing (absdiff between consecutive frames) detects change cheaply. MOG2 and KNN subtractors learn a per-pixel background model over time and return a foreground mask (MOG2 marks shadows as 127 when detectShadows=True). Morphology cleans the mask before contour analysis.

**Use when:** Detect moving objects with a static camera, e.g. people entering a monitored zone.

**Pitfalls:** Needs a static camera and warm-up frames. Lighting changes and moving backgrounds (trees, screens) cause false foreground. Threshold away shadow value 127 if shadows are not wanted.

```python
sub = cv.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
frames = [image] * 5 + [cv.rectangle(image.copy(), (20, 20), (80, 80), (255, 255, 255), -1)]
for frame in frames:
    fgmask = sub.apply(frame)
_, fgmask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)   # drop shadows (127)
fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
diff = cv.absdiff(cv.cvtColor(frames[-2], cv.COLOR_BGR2GRAY), cv.cvtColor(frames[-1], cv.COLOR_BGR2GRAY))
_, motion = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
```

Sources: supplement:bgsub.

## Optical flow (Lucas-Kanade and Farneback)

Optical flow estimates apparent motion between frames assuming brightness constancy. Lucas-Kanade (sparse) tracks selected points, usually from goodFeaturesToTrack, using image pyramids. Farneback (dense) returns a 2-channel (dx, dy) vector for every pixel, often visualized as HSV with angle as hue and magnitude as value.

**Use when:** Track points across frames, measure motion direction and speed.

**Pitfalls:** Points must be float32 of shape (N,1,2). Keep only status==1 results. Large motion, occlusion and lighting change break the brightness-constancy assumption; use pyramids (maxLevel) for larger motion.

```python
prev = g
M = np.float32([[1, 0, 3], [0, 1, 2]])
nxt = cv.warpAffine(g, M, (g.shape[1], g.shape[0]))
p0 = cv.goodFeaturesToTrack(prev, 50, 0.01, 7)
if p0 is not None:
    p1, st, err = cv.calcOpticalFlowPyrLK(prev, nxt, p0, None, winSize=(15, 15), maxLevel=2)
    good_new, good_old = p1[st == 1], p0[st == 1]
flow = cv.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
```

Sources: supplement:optflow.

## Histogram back-projection, mean-shift and CamShift tracking

Back-projection replaces each pixel with the probability of its color under a target histogram (usually hue). meanShift moves a window to the local density peak of that map; CamShift also adapts window size and orientation. compareHist scores similarity between two histograms (correlation: higher is better; Bhattacharyya: lower is better).

**Use when:** Track a colored object or compare color distributions of regions.

**Pitfalls:** Hue is unreliable for very dark or unsaturated pixels; mask them with inRange on S and V. Hue range in OpenCV is 0..179. Histograms passed to compareHist must have the same size and type.

```python
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
x, y, w, h = 40, 40, 60, 60
roi = hsv[y:y + h, x:x + w]
roi_hist = cv.calcHist([roi], [0], None, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
back = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
term = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
_, window = cv.meanShift(back, (x, y, w, h), term)
rot_box, window = cv.CamShift(back, (x, y, w, h), term)
h2 = cv.calcHist([hsv], [0], None, [180], [0, 180])
similarity = cv.compareHist(roi_hist, h2, cv.HISTCMP_CORREL)
```

Sources: supplement:meanshift.

## LAB, YCrCb and HLS color spaces

HSV separates hue from saturation/value. LAB approximates perceptual uniformity: L is lightness, a is green-red, b is blue-yellow, so Euclidean distance in LAB better reflects visible color difference. YCrCb separates luma (Y) from chroma (Cr, Cb) and is common for skin detection and for equalizing only brightness.

**Use when:** Pick a color space where the property you threshold or compare is isolated.

**Pitfalls:** 8-bit LAB is scaled (L 0..255, a/b offset by 128), unlike textbook ranges. Skin thresholds are heuristic and vary across people and lighting. Convert back to BGR before display.

```python
lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
L, A, B = cv.split(lab)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = cv.cvtColor(cv.merge([clahe.apply(L), A, B]), cv.COLOR_LAB2BGR)
ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
skin = cv.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
ref = np.array([[[0, 0, 255]]], np.uint8)
ref_lab = cv.cvtColor(ref, cv.COLOR_BGR2LAB).astype(np.float32)
dist = np.linalg.norm(lab.astype(np.float32) - ref_lab, axis=2)
```

Sources: supplement:colorspaces.

## Morphological skeletons and thinning

A morphological skeleton reduces a binary shape to a one-pixel-wide centerline by repeatedly eroding and collecting the residue of an opening. It preserves topology approximately and is useful for measuring strokes, roads or cracks. The hit-or-miss transform finds exact local patterns.

**Use when:** Measure thin structures, analyze handwriting or vessel/crack networks.

**Pitfalls:** Noise creates spurious branches; clean the mask first. This simple skeleton can be disconnected; dedicated thinning (opencv-contrib ximgproc.thinning) gives connected results but is not in the core package.

```python
_, bw = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
skel = np.zeros_like(bw)
element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
work = bw.copy()
while cv.countNonZero(work) > 0:
    opened = cv.morphologyEx(work, cv.MORPH_OPEN, element)
    skel = cv.bitwise_or(skel, cv.subtract(work, opened))
    work = cv.erode(work, element)
```

Sources: supplement:morphology.

## Stereo disparity, epipolar geometry and depth

With two rectified cameras a point shifts horizontally between views; that shift is disparity d. Depth Z = f*B/d (focal length in pixels times baseline). StereoBM/StereoSGBM compute dense disparity. For unrectified views, the fundamental matrix F relates corresponding points via x2^T F x1 = 0 and defines epipolar lines.

**Use when:** Estimate depth from two calibrated cameras or verify correspondences geometrically.

**Pitfalls:** StereoBM requires rectified 8-bit grayscale pairs; numDisparities must be divisible by 16 and blockSize odd. Output is fixed-point (divide by 16). Textureless areas give invalid disparity. Depth needs real calibration values.

```python
shift = np.float32([[1, 0, -8], [0, 1, 0]])
right = cv.warpAffine(g, shift, (g.shape[1], g.shape[0]))
stereo = cv.StereoBM_create(numDisparities=32, blockSize=15)
disparity = stereo.compute(g, right).astype(np.float32) / 16.0
f_px, baseline_m = 700.0, 0.12
valid = disparity > 0
depth = np.zeros_like(disparity)
depth[valid] = f_px * baseline_m / disparity[valid]
```

Sources: supplement:stereo.

## Running pretrained neural networks with cv.dnn

OpenCV dnn runs pretrained models (ONNX, Caffe, TensorFlow, Darknet) on CPU without a training framework. blobFromImage resizes, scales, mean-subtracts, optionally swaps BGR to RGB and returns an NCHW float blob. The model weights are separate files that must be obtained and stored locally for offline use.

**Use when:** Use a CNN classifier/detector/segmenter when handcrafted features are insufficient, still fully offline.

**Pitfalls:** Preprocessing (size, scale, mean, channel order) must match how the model was trained. No weights are bundled in this knowledge base. Large models may exceed the project size budget.

```python
blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(224, 224), mean=(0, 0, 0), swapRB=True, crop=False)
assert blob.shape == (1, 3, 224, 224)
# With a downloaded model file:
# net = cv.dnn.readNetFromONNX('model.onnx')
# net.setInput(blob)
# scores = net.forward()
```

Sources: supplement:dnn.

## Matplotlib figures for image analysis

Matplotlib figures compare processing stages. Use subplots for grids, set cmap="gray" with vmin/vmax for single-channel images, add colorbar for signed or float maps (e.g. gradients, distance transform), and plot 1D intensity profiles or histograms with labelled axes and legends. savefig writes a file; show opens a window.

**Use when:** Produce clear, labelled visual evidence for lab reports and debugging.

**Pitfalls:** Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable. Close figures in loops to free memory. plt.show blocks in scripts and does nothing with the Agg backend.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sx = cv.Sobel(g, cv.CV_64F, 1, 0, ksize=3)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)); axes[0].set_title('Original')
im = axes[1].imshow(sx, cmap='seismic'); axes[1].set_title('Sobel x (signed)')
fig.colorbar(im, ax=axes[1])
axes[2].plot(g[g.shape[0] // 2], label='middle row'); axes[2].set_xlabel('x'); axes[2].legend()
for a in axes[:2]: a.axis('off')
fig.suptitle('Processing stages'); fig.tight_layout()
fig.savefig('stages.png', dpi=100); plt.close(fig)
```

Sources: supplement:matplotlib.
