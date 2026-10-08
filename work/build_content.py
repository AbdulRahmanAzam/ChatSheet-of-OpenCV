from pathlib import Path
import json, re, ast, hashlib, sqlite3, html
ROOT=Path(__file__).resolve().parents[1]/'outputs'/'cv_knowledge_base'
topics=[]
def topic(id,title,aliases,answer,when,code,pitfalls,source,related=''):
    topics.append(dict(id=id,title=title,aliases=aliases.split(';') if aliases else [],
        answer=answer,when_to_use=when,code=code,pitfalls=pitfalls,
        source_refs=source.split(';') if source else [],related=related.split(';') if related else [],
        provenance='Authored explanation and corrected example; source refs indicate topic grounding, not verbatim quotation.',
        scope='core',review_status='curated',code_context='import cv2 as cv; import numpy as np; image is decoded BGR uint8; g is grayscale uint8 unless stated'))

topic('pixels','Pixels, resolution, coordinates and channels','image shape;rows;columns;image depth;channels',
      'An image is a sampled array. A grayscale image has shape (height,width); BGR color has (height,width,3). Pixel depth means the numeric type/precision, while geometric depth is distance in a 3D scene. Resolution controls spatial sampling, not semantic detail recovered from nothing.',
      'Before indexing, cropping, resizing, or interpreting ddepth.', 'height, width = image.shape[:2]\npixel = image[y, x]\nblue, green, red = image[y, x]',
      'NumPy indexes row then column; OpenCV drawing points are (x,y). uint8 arithmetic can wrap. Pixel coordinate origin is top-left.', 'lab_01_manual:p10','numpy;depth;coordinates')
topic('io','Read and save an image','imread;imwrite;load image;image file',
      'Decode the file into an array and check for failure before using its shape or converting colors. Standard color reads are BGR. Grayscale reads request a single channel. Saving an array encodes it according to the output filename extension.',
      'Start any image-processing exercise.',"image = cv.imread('input.png', cv.IMREAD_COLOR)\nif image is None:\n    raise FileNotFoundError('input.png')\nif not cv.imwrite('output.png', image):\n    raise OSError('Could not save image')",
      'A path can exist but contain an unsupported/corrupt image. Current working directory affects relative paths. Do not silently replace missing course input with synthetic data.', 'lab_01_manual:p14','colors;display')
topic('display','Plot images correctly','imshow;matplotlib;plt.imshow;BGR to RGB;subplot;subplots',
      'OpenCV color arrays normally use BGR, while Matplotlib interprets a three-channel array as RGB. Convert only for plotting. For comparable grayscale panels set a fixed range. plt.subplots returns a figure and axes, which is convenient for labeled comparisons.',
      'Inspect processing stages and export a reproducible figure.',"import matplotlib.pyplot as plt\nfig, ax = plt.subplots(1, 2, figsize=(10, 4))\nax[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))\nax[1].imshow(g, cmap='gray', vmin=0, vmax=255)\nfor a in ax: a.axis('off')\nfig.tight_layout()\nfig.savefig('comparison.png')\nplt.close(fig)",
      'Repeated BGR/RGB conversion swaps colors back. imshow on float RGB expects values near 0..1. cv.imshow needs a GUI-enabled OpenCV build; saved Matplotlib figures work headlessly.', 'lab_01_manual:p14;lab_manual_06:p10','colors;histogram')
topic('colors','Color conversion and HSV ranges','cvtColor;color conversion;BGR;RGB;HSV;grayscale;VGA',
      'Grayscale combines channels into luminance-like intensity for many geometric operations. HSV separates hue, saturation, and value for color masks. VGA describes a display resolution/standard, not a color space; the intended operation in the earlier example is BGR to RGB.',
      'Use grayscale for thresholding/gradients; HSV for color-based selection.', 'g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)\nrgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)\nhsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)',
      'For OpenCV uint8 HSV, hue is 0..179 and saturation/value are 0..255. Floating HSV uses different scaling. A red hue range often wraps and needs two masks.', 'lab_01_manual:p15;lab_05_manual:p9','mask;display')
topic('numpy','NumPy arrays and safe image arithmetic','np.sqrt;np.hypot;np.clip;np.uint8;np.float32;reshape;ravel;flatten;zeros;ones;astype',
      'NumPy stores images and performs vectorized arithmetic. Convert to floating point before squaring, subtracting, or averaging when intermediate values exceed uint8. reshape changes layout without changing element order; ravel may return a view. Clip and round before converting results back to uint8.',
      'Gradient magnitude, pixel clustering, feature vectors, masks, and normalization.', 'f = g.astype(np.float32)\nsquared = f*f\nmagnitude = np.hypot(dx, dy)\nresult = np.clip(np.rint(f), 0, 255).astype(np.uint8)\npixels = image.reshape(-1, 3).astype(np.float32)',
      'np.uint8([-1]) or a cast of out-of-range floats does not provide meaningful saturation. Squaring uint8 before conversion has already lost information.', 'lab_04_manual:p14;lab_manual_06:p10','depth;gradients')
topic('depth','Output depth and signed responses','ddepth;CV_32F;CV_64F;CV_16S;-1',
      'ddepth chooses the output numeric depth. -1 keeps the input depth, which is suitable for some smoothing but clips negative derivative responses when the input is unsigned. CV_32F or CV_64F preserves negative gradients and large intermediate results.',
      'Sobel, Scharr, Laplacian, filter2D, texture statistics.', 'dx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)\ndisplay = cv.convertScaleAbs(dx)',
      'Output depth does not mean neural-network depth or physical distance. An absolute display image loses the sign needed for orientation and zero crossings.', 'lab_manual_06:p8;lab_04_manual:p17','gradients;convert_abs')
topic('crop','Cropping and region of interest','ROI;crop;center crop;R10',
      'A region of interest is a rectangular or masked subset of an image. Slicing uses [y_start:y_end,x_start:x_end], with exclusive end indices. A center crop of size s starts at ((width-s)//2,(height-s)//2). R10 is unresolved terminology in the user request; it is not automatically treated as an API.',
      'Inspect fine detail or limit processing to a relevant area.', 'h, w = image.shape[:2]\ns = 300\nif min(h,w) < s: raise ValueError("image too small")\nx, y = (w-s)//2, (h-s)//2\nroi = image[y:y+s, x:x+s].copy()',
      'A slice may be a view: modifying it can modify the original image. Clamping coordinates silently can change the requested crop size.', 'lab_01_manual:p18;lab_01_tasks:p2','coordinates;mask')
topic('resize','Resize and interpolation','resize;INTER_AREA;INTER_LINEAR;INTER_NEAREST;INTER_CUBIC',
      'Resizing resamples the image to a different pixel grid. INTER_AREA is often useful for shrinking, linear/cubic for natural-image enlargement, and nearest neighbor for label masks. A larger output does not recover lost fine structure.',
      'Standardize input dimensions or show a scaled image.', 'small = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)\nmask2 = cv.resize(mask, (500, 500), interpolation=cv.INTER_NEAREST)',
      'dsize is (width,height). Resizing two medical images to the same size does not register their anatomy.', 'lab_01_manual:p16;lab_03_manual:p6','affine;registration')
topic('drawing','Draw circles, rectangles, lines and text','circle;rectangle;line;putText;drawing canvas',
      'Draw into a writable array using pixel positions and BGR color tuples. A negative thickness fills supported shapes. Text needs an origin on the baseline, a font, scale, color, and thickness.',
      'Annotate detections, create masks, or draw a synthetic test image.', 'canvas = np.zeros((800,800,3), np.uint8)\ncv.circle(canvas, (400,400), 100, (0,255,0), 2)\ncv.putText(canvas, "screen", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)',
      'Drawing mutates the array. For an 800x800 grid, the geometric center of pixel centers is (399.5,399.5); (400,400) is the chosen integer drawing center.', 'lab_01_tasks:p2;lab_01_manual:p19','pixels;mask')
topic('blend','Weighted blending versus image addition','add;addWeighted;alpha blend;overlay',
      'cv.add adds intensities with saturation for uint8. Alpha blending uses weighted inputs and an optional offset. To add a transparent banner, copy the image, draw the banner on the copy, then blend it with the original.',
      'Overlays, registered image fusion, gradual transitions.', 'overlay = image.copy()\noverlay[int(image.shape[0]*0.8):] = (255,0,0)\nout = cv.addWeighted(overlay, 0.4, image, 0.6, 0)',
      'Addition is not a 50/50 blend. Inputs need matching size/type. Register images before interpreting anatomical or geometric correspondence.', 'lab_01_manual:p23;lab_02_tasks:p5','registration;mask')
topic('mask','Binary masks and bitwise composition','inRange;bitwise_and;bitwise_or;bitwise_not;masking functions',
      'A mask selects locations. A uint8 single-channel mask commonly uses 0 for excluded and 255 for included. inRange produces such a mask from channel bounds. Complementary masks can choose foreground pixels from one image and background pixels from another.',
      'Color extraction, object cutouts, ROI selection, compositing.', 'mask = cv.inRange(hsv, np.array([20,80,80]), np.array([35,255,255]))\nselected = cv.bitwise_and(image, image, mask=mask)\ninverse = cv.bitwise_not(mask)',
      'Mask dimensions must match the image. AND/OR on arbitrary intensities are bit operations, not probabilistic blending.', 'lab_01_tasks:p3;lab_05_manual:p9','colors;morphology')
topic('histogram','Image histograms and bar plots','calcHist;np.histogram;plt.hist;plt.bar;plot dot bar;edges histogram',
      'A histogram counts samples in bins; it discards spatial location. Intensity histograms summarize brightness, color histograms summarize channel distributions, and edge-direction histograms summarize orientation. Histogram bin edges are not image edges.',
      'Inspect contrast, choose thresholds, compare distributions, or form descriptors.', 'hist, edges = np.histogram(g, bins=256, range=(0,256))\nimport matplotlib.pyplot as plt\nplt.bar(edges[:-1], hist, width=1)\nplt.xlabel("Intensity"); plt.ylabel("Pixel count")',
      'Specify range and normalization consistently. Different images can have the same histogram. Count histograms and density histograms have different units.', 'lab_04_manual:p9;lab_05_tasks:p3','equalization;hed')
topic('equalization','Histogram equalization and CLAHE','equalizeHist;CLAHE;createCLAHE;contrast enhancement',
      'Histogram equalization remaps grayscale intensities using their cumulative distribution to spread occupied values. CLAHE adapts locally with a contrast limit. Neither creates missing image information, and noise may become more visible.',
      'Improve visible contrast when the input occupies a narrow intensity range.', 'equalized = cv.equalizeHist(g)\nclahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\nlocal = clahe.apply(g)',
      'equalizeHist expects uint8 single-channel input. Independently equalizing BGR channels alters colors; work on a luminance channel when color fidelity matters.', 'lab_01_manual:p24;lab_02_tasks:p4','histogram;gamma')
topic('gamma','Log and gamma intensity transforms','log transform;gamma correction;power law;np.log1p',
      'Using output=255*(input/255)^gamma, gamma below one brightens midtones and gamma above one darkens them. A logarithmic transform expands dark values and compresses highlights. Both are point operations without spatial understanding.',
      'Visualize low-intensity detail or compress a display range.', 'f = g.astype(np.float32)/255\nout = np.clip(255*f**0.6, 0, 255).astype(np.uint8)\nlog_image = (255*np.log1p(g.astype(np.float32))/np.log(256)).astype(np.uint8)',
      'Some libraries define a reciprocal gamma parameter; state the convention. Convert before arithmetic. Pseudocolor and display transforms are not measurements of disease or flow.', 'lab_02_tasks:p4;lab_02_tasks:p6','equalization;pseudocolor')
topic('pseudocolor','Pseudocolor and intensity visualization','applyColorMap;JET;HOT;heatmap;color balance',
      'A colormap assigns display colors to scalar intensities. It can make differences easier to see but adds no measured spectral, anatomical, or flow information. Channel gains applied afterward are a visualization choice.',
      'The Lab 02 grayscale imaging demonstrations.', 'heatmap = cv.applyColorMap(g, cv.COLORMAP_JET)',
      'JET colors in an echocardiogram are not Doppler measurements. Keep the original scalar image available and avoid interpreting chosen colors as clinical labels.', 'lab_02_tasks:p4;lab_02_tasks:p6','gamma;blend')
topic('kernel','Kernels, correlation and convolution','filter;kernel;filter2D;convolution;correlation',
      'A kernel assigns weights to a local neighborhood. Correlation multiplies the neighborhood by the kernel directly; convolution flips the kernel along both axes first. Symmetric blur kernels give the same result either way. OpenCV filter2D performs correlation.',
      'Smoothing, sharpening, derivatives, and custom local responses.', 'k = np.ones((3,3),np.float32)/9\nsmooth = cv.filter2D(g, cv.CV_32F, k)\ntrue_convolution = cv.filter2D(g, cv.CV_32F, k[::-1,::-1])',
      'A negative/signed kernel with uint8 output clips information. Normalize averaging kernels. The manual incorrectly treats all filter2D examples as different convolution sizes.', 'lab_04_manual:p16;lab_04_manual:p18','padding;depth;box')
topic('padding','Full, same, valid, stride and border handling','padding;valid convolution;same convolution;full convolution;strides;border types;BORDER_REFLECT',
      'For odd kernel size k and stride one, valid output size is n-k+1, same is n, and full is n+k-1. Border rules supply values outside the image: constant, replicate, reflect including the edge, or reflect101 excluding the edge. Output shape and border rule are separate choices.',
      'Reproduce textbook convolution dimensions or reason about edge artifacts.', "from cv_core import convolution\nvalid = convolution(g, k, mode='valid', stride=2)\npadded = cv.copyMakeBorder(g, 2,2,2,2, cv.BORDER_REFLECT_101)",
      'filter2D has no strides keyword and always keeps spatial size. BORDER_CONSTANT alone does not make valid convolution. Unsupported border modes vary by function; do not assume WRAP is accepted everywhere.', 'lab_04_manual:p18;lab_04_manual:p19','kernel;gaussian')
topic('box','Box blur','box filter;botch filter;blur;boxFilter;average filter',
      'A normalized box kernel averages all pixels in a rectangular neighborhood equally. It reduces small variations but softens boundaries. The supplied manual calls this Box Blur; botch filter appears to be a spelling/transcription ambiguity, not a separate verified operator.',
      'Fast simple smoothing or local means.', 'blurred = cv.blur(image, (5,5))\nmean = cv.boxFilter(g.astype(np.float32), -1, (5,5), normalize=True)',
      'An unnormalized box filter returns a sum, which can overflow or saturate. It is not a local standard-deviation calculation.', 'lab_04_manual:p16','gaussian;texture')
topic('gaussian','Gaussian blur and Gaussian kernels','GaussianBlur;getGaussianKernel;sepFilter2D;sigma;Gaussian smoothing',
      'Gaussian smoothing weights nearby pixels most strongly and suppresses high-frequency variation. sigma controls the spread; the odd kernel size truncates its support. A 2D Gaussian is separable into horizontal and vertical 1D filtering.',
      'Denoise before gradients, Canny, or scale-space analysis.', 'smooth = cv.GaussianBlur(image, (5,5), 1.0)\nv = cv.getGaussianKernel(5, 1.0)\nk2d = v @ v.T\nequivalent = cv.sepFilter2D(image, -1, v, v)',
      'getGaussianKernel returns one column, so passing it alone to filter2D smooths only one direction. Larger sigma removes more detail.', 'lab_01_manual:p17;lab_04_manual:p17','kernel;canny')
topic('median','Median filtering','medianBlur;median blur;salt and pepper noise',
      'A median filter selects the median value in a neighborhood rather than averaging it. It is useful for isolated high/low impulse noise and can preserve sharp transitions better than linear averaging.',
      'Salt-and-pepper noise or preprocessing for some circle-detection tasks.', 'smooth = cv.medianBlur(image, 5)',
      'Kernel size must be odd and greater than one. Large windows remove small features. Supported input depths depend on kernel size.', 'lab_06_tasks:p3','gaussian;circles')
topic('bilateral','Bilateral filtering','bilateralFilter;edge preserving smoothing',
      'Bilateral filtering combines spatial proximity and similarity in intensity/color. Nearby pixels across a large color discontinuity contribute less, allowing some smoothing within regions while retaining boundaries.',
      'Denoising when preserving significant intensity boundaries matters.', 'smooth = cv.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)',
      'Usually slower than a small Gaussian filter. Large range sigma makes it behave more like spatial smoothing; it can flatten subtle texture that a later descriptor needs.', 'supplement:filtering','gaussian;texture')
topic('gradients','Sobel derivatives and gradient magnitude','Sobel;Sobel-Feldman;sobel_x;sobel_y;gradient;np.sqrt',
      'Sobel estimates first spatial derivatives using a small derivative kernel with smoothing in the perpendicular direction. dx responds to left-right intensity change, often revealing vertical boundaries; dy reveals horizontal ones. Magnitude combines them; atan2 gives signed orientation.',
      'Edges, orientation histograms, HOG, and geometric boundary cues.', 'dx = cv.Sobel(g, cv.CV_32F, 1,0,ksize=3)\ndy = cv.Sobel(g, cv.CV_32F, 0,1,ksize=3)\nmagnitude = np.hypot(dx,dy)\nangle = np.degrees(np.arctan2(dy,dx))',
      'Sobel-Feldman is another name for Sobel, not a separate diagonal detector. A derivative response is not yet a thin binary edge map.', 'lab_04_manual:p20;lab_04_manual:p21;lab_manual_06:p6','depth;hed;canny')
topic('scharr','Scharr and Prewitt operators','Scharr;SCAR;Prewitt;edge detection',
      'Scharr uses derivative weights designed to improve rotational accuracy of a 3x3 approximation. Prewitt uses simpler uniform smoothing weights. These yield signed derivatives, which can be combined into gradient magnitude.',
      'Compare compact gradient operators; Scharr is useful when a small kernel is required.', 'sx = cv.Scharr(g, cv.CV_32F, 1,0)\nsy = cv.Scharr(g, cv.CV_32F, 0,1)\nprewitt_x = cv.filter2D(g, cv.CV_32F, np.float32([[-1,0,1],[-1,0,1],[-1,0,1]]))',
      'SCAR in the request corresponds to the Scharr topic present in the manual. Derivative scales differ; the same threshold is not automatically comparable across operators.', 'lab_04_manual:p17;lab_04_manual:p20','gradients;depth')
topic('laplacian','Laplacian and Laplacian of Gaussian','Laplacian;LoG;second derivative;zero crossing',
      'The Laplacian sums second spatial derivatives. Smoothing first reduces noise sensitivity, producing a discrete Laplacian-of-Gaussian style pipeline. Edge localization by zero crossings uses changes in the sign of the response, often with a minimum response-range threshold.',
      'Detect changes in gradient or explore second-derivative edges.', 'blur = cv.GaussianBlur(g,(5,5),1.4)\nresponse = cv.Laplacian(blur,cv.CV_32F)\nvisualization = cv.convertScaleAbs(response)',
      'The absolute response visualization is not a zero-crossing edge map. Keep signed float response for zero-crossing tests.', 'lab_manual_06:p14;lab_manual_06:p15;lab_manual_06:p16','depth;convert_abs')
topic('convert_abs','Absolute scaling for visualization','convertScaleAbs;convert scale absolute',
      'convertScaleAbs applies scaling and offset, takes absolute values, and saturates into uint8. It is convenient for displaying signed gradients, but destroys polarity and clips values outside the display range.',
      'Show a Sobel or Laplacian response after keeping its signed original.', 'view = cv.convertScaleAbs(dx, alpha=1, beta=0)',
      'Do not compute orientation or zero crossings from the absolute uint8 image.', 'lab_manual_06:p15','depth;laplacian')
topic('canny','Canny edge detection','Canny;hysteresis;non maximum suppression;strong weak edges',
      'A Canny pipeline smooths noise, computes gradients, thins local maxima along the gradient direction, then retains strong edges and weak edges connected to them. Low and high thresholds govern hysteresis. The result is a thin binary edge map, not filled object regions.',
      'Prepare line detection or inspect boundaries.', 'blur = cv.GaussianBlur(g,(5,5),1.4)\nedges = cv.Canny(blur,50,150,L2gradient=True)',
      'Use explicit blur to control smoothing. Non-maximum suppression is not simply a weak-pixel threshold. Final Canny output alone does not label which surviving pixels began strong or weak.', 'lab_04_manual:p22;lab_manual_06:p11;lab_manual_06:p13','gradients;hough;threshold')
topic('threshold','Global thresholding and threshold types','threshold;THRESH_BINARY;THRESH_BINARY_INV;THRESH_TRUNC;TOZERO',
      'A global threshold applies one cutoff everywhere. Binary output chooses zero or maxval; inverse binary swaps those assignments. Truncation caps high values, and to-zero keeps only values on a selected side. Global binary thresholding works best with separable foreground/background intensities.',
      'Simple segmentation under reasonably uniform illumination.', 'value, mask = cv.threshold(g,127,255,cv.THRESH_BINARY)\ninverted = cv.threshold(g,127,255,cv.THRESH_BINARY_INV)[1]',
      'maxval=200 means intensity200, not255. A plotting program may rescale it to white. Uneven light can defeat any single cutoff.', 'lab_01_manual:p21;lab_05_manual:p8','adaptive;otsu')
topic('adaptive','Adaptive mean and Gaussian thresholding','adaptiveThreshold;blockSize;C;adaptive mean;adaptive Gaussian',
      'Adaptive thresholding computes a local mean or Gaussian-weighted mean and subtracts C. Each pixel is compared with that local threshold. Small neighborhoods follow fine variation but can fragment strokes; very large ones approach broad illumination averaging.',
      'Documents with shadows or spatially varying brightness.', 'mask = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,7)',
      'blockSize must be odd and greater than1, with uint8 single-channel input. Increasing C lowers the threshold: more white pixels for BINARY and fewer for BINARY_INV.', 'lab_05_manual:p8;lab_05_manual:p9;lab_05_tasks:p2','threshold;otsu')
topic('otsu','Otsu automatic threshold selection','Otsu;THRESH_OTSU;bimodal histogram',
      'Otsu chooses a global threshold by optimizing the separation of two intensity classes. It uses the image histogram and returns the selected numeric threshold as well as the binary mask. It can work well when foreground/background form distinct groups.',
      'Automatically segment high-contrast grayscale objects under fairly uniform illumination.', 't, mask = cv.threshold(g,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)',
      'Otsu does not fix spatially varying illumination or guarantee semantic objects. Blur/contrast transforms change the histogram and can change the selected threshold.', 'lab_05_manual:p9;lab_05_tasks:p3','histogram;adaptive')
topic('morphology','Morphological erosion, dilation, opening and closing','morphological operations;erode;dilate;opening;closing;structuring element',
      'On a binary foreground mask, dilation expands foreground and erosion shrinks it according to a structuring element. Opening is erosion then dilation, removing small foreground details. Closing is dilation then erosion, filling small gaps. Neighborhood shape and iteration count control the spatial scale.',
      'Clean threshold masks, join broken regions, and prepare watershed markers.', 'k = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))\nopened = cv.morphologyEx(mask,cv.MORPH_OPEN,k)\nclosed = cv.morphologyEx(mask,cv.MORPH_CLOSE,k)',
      'Foreground polarity matters. Large kernels can remove real small objects or merge neighbors. Repeated operations change geometry.', 'lab_05_manual:p13;lab_05_tasks:p7','distance;watershed')
topic('morphology_more','Morphological gradient, top-hat and black-hat','MORPH_GRADIENT;MORPH_TOPHAT;MORPH_BLACKHAT',
      'The morphological gradient is dilation minus erosion, highlighting a band around boundaries. Top-hat is the original minus its opening, emphasizing small bright structures. Black-hat is closing minus the original, emphasizing small dark structures.',
      'Boundary visualization and simple uneven-background correction.', 'k = cv.getStructuringElement(cv.MORPH_RECT,(15,15))\nbright = cv.morphologyEx(g,cv.MORPH_TOPHAT,k)\ndark = cv.morphologyEx(g,cv.MORPH_BLACKHAT,k)',
      'Choose an element larger than the structures being separated from the background. These are morphological operations, not arbitrary linear convolution kernels.', 'supplement:morphology','morphology;adaptive')
topic('region','Seeded region growing','region growing;flood fill;seed;connectivity',
      'Start at a seed and visit connected neighbors that satisfy a similarity rule. The supplied solution compares each candidate with the fixed seed intensity, avoiding accidental drift of an evolving mean. Four-connectivity uses axial neighbors; eight-connectivity also uses diagonals.',
      'Select one connected fairly uniform region when a seed is known.', 'from cv_core import region_grow\nmask = region_grow(image, seed=(100,80), tolerance=15, connectivity=4)',
      'Seed order is (x,y) here; the manual uses array-style coordinates. Convert intensity values to int before subtraction. Wrong seed or tolerance leaks or undersegments. Fixed seed and evolving-mean algorithms differ.', 'lab_05_manual:p10;lab_05_manual:p11','threshold;components')
topic('distance','Distance transforms and foreground seeds','distanceTransform;DIST_L2;sure foreground',
      'A distance transform gives each nonzero pixel its distance to the nearest zero pixel. Thresholding high distances finds interior regions of sufficiently thick foreground objects, often useful as watershed seeds.',
      'Create markers inside blobs and separate touching objects.', 'dist = cv.distanceTransform(mask,cv.DIST_L2,5)\nsure_fg = np.uint8(dist > 0.5*dist.max())*255',
      'The input must have the correct foreground polarity and a surrounding zero background. A high fraction can erase small-object seeds; a low fraction can merge touching-object seeds.', 'lab_05_manual:p13;lab_05_tasks:p8','watershed;morphology')
topic('components','Connected components and integer labels','connectedComponents;connectedComponentsWithStats;markers',
      'Connected-component labeling gives each connected nonzero region an integer label; zero denotes background. The returned count includes background. Stats can report bounding boxes and pixel areas.',
      'Count separated mask blobs, form watershed seeds, or filter small regions.', 'count, labels, stats, centers = cv.connectedComponentsWithStats(mask, connectivity=8)\nobject_count = count-1',
      'Touching objects can share one component. A component count is not necessarily a real-object count.', 'lab_05_manual:p13','distance;contours')
topic('watershed','Marker-controlled watershed segmentation','watershed;markers;unknown;touching objects',
      'Provide labeled confident interiors and an unknown area. Watershed propagates labels through image variation, marking separating boundaries. The teaching pipeline uses thresholding, opening, dilated background support, distance-based foreground, connected components, and unknown pixels.',
      'Separate touching objects when useful seeds can be created.', 'from cv_core import watershed_stages\nstages = watershed_stages(image, distance_fraction=0.5, foreground="bright")\nlabels = stages["markers"]',
      'cv.watershed takes a uint8 3-channel image and int32 markers. Unknown starts at0, boundaries become-1. In this pipeline background is1. It mutates markers. Poor seeds cause poor segmentation.', 'lab_05_manual:p12;lab_05_manual:p13;lab_05_tasks:p7','morphology;distance;components')
topic('kmeans','K-means color clustering','kmeans;clustering;compactness;centers;labels',
      'K-means alternates assigning pixels to the nearest center and updating centers, minimizing within-cluster squared distances. Reshape color pixels to N-by-3 float32, cluster, replace each pixel by its center, and reshape back.',
      'Color quantization and an unsupervised segmentation baseline.', 'from cv_core import segment_kmeans\nsegmented, labels, compactness = segment_kmeans(image, k=4, seed=42)',
      'Color clusters are not guaranteed semantic objects or connected regions. K controls model size, and initialization can affect results. Evaluate K choices instead of assuming larger is always better.', 'lab_05_manual:p14;lab_05_manual:p15;lab_05_tasks:p9','numpy;segmentation')
topic('segmentation','Segmentation versus edges, detection and classification','image segmentation;semantic segmentation;instance segmentation;classification;object detection',
      'Segmentation assigns labels to pixels; detection locates instances, often with boxes; classification assigns a category to an image or region. Semantic segmentation gives class labels, while instance segmentation distinguishes individual objects. An edge map only identifies local boundaries.',
      'Choose the actual output required before choosing an algorithm.', 'mask = cv.threshold(g,127,255,cv.THRESH_BINARY)[1]',
      'A color cluster or threshold region does not inherently know whether it is a screen, lesion, or coin. Labels need assumptions, reference data, or a trained classifier.', 'lab_01_manual:p5;lab_05_manual:p3;lab_05_manual:p5','threshold;kmeans;contours')
topic('contours','Contours and geometric boundaries','findContours;approxPolyDP;boundingRect;contourArea;perimeter',
      'Contours trace boundaries in a binary mask. Their area, perimeter, polygon approximation, and bounding boxes support shape analysis. Polygon approximation can reduce a quadrilateral boundary to four vertices, but the semantic identity still needs context.',
      'Find object outlines and candidate screen/document boundaries.', 'contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)\nfor contour in contours:\n    polygon = cv.approxPolyDP(contour,0.02*cv.arcLength(contour,True),True)',
      'RETR_EXTERNAL ignores nested holes. A bounding rectangle is not a perspective quadrilateral; contours and bounding-box pixel areas differ.', 'supplement:contours;lab_06_tasks:p1','components;hough;perspective')
topic('coordinates','Homogeneous coordinates and matrix composition','2x2 matrix;3x3 matrix;homogeneous coordinates;matrix multiplication',
      'Use [x,y,1] to represent 2D points so translation and linear transforms compose as 3x3 matrices. With column vectors, the rightmost matrix acts first. Divide by the third coordinate after a projective transform.',
      'Combine scale, rotation, translation, shear, and perspective in a single resampling.', 'p = np.array([x,y,1.0])\nq = H @ p\nq = q[:2]/q[2]',
      'Image y increases downward. A standard Cartesian positive-angle matrix therefore appears clockwise on an image. Avoid repeated image warps when matrices can be composed.', 'lab_03_manual:p15;lab_03_manual:p17','affine;perspective')
topic('affine','Linear, rigid, similarity and affine transformations','affine transform;rigid;similarity;shear;translation;scaling;rotation',
      'A linear 2D map uses a 2x2 matrix and fixes the origin. A rigid map adds rotation and translation while preserving distances. Similarity adds uniform scaling. An affine map additionally allows nonuniform scaling and shear and preserves parallel lines. Three noncollinear point pairs determine a 2D affine map.',
      'Geometric correction when perspective effects are absent or negligible.', 'src = np.float32([[0,0],[100,0],[0,100]])\ndst = np.float32([[20,10],[120,20],[10,110]])\nM = cv.getAffineTransform(src,dst)\nwarped = cv.warpAffine(image,M,(image.shape[1],image.shape[0]))',
      'Affine does not preserve every length or angle. Three collinear points are degenerate. Same-size output can clip transformed content.', 'lab_03_manual:p5;lab_03_manual:p11;lab_03_manual:p13;lab_03_tasks:p2','coordinates;perspective')
topic('perspective','Perspective transformations and homographies','homography;getPerspectiveTransform;warpPerspective;four corners;projective transform',
      'A 3x3 homography maps between views of a plane or images related by pure camera rotation. It preserves straight lines but can change angles, length ratios, and parallelism. Four nondegenerate point pairs determine it up to scale.',
      'Rectify documents/paintings, map a planar reference into a scene, or stitch suitable views.', 'src = np.float32([[10,20],[300,15],[310,220],[20,230]])\ndst = np.float32([[0,0],[399,0],[399,399],[0,399]])\nH = cv.getPerspectiveTransform(src,dst)\nwarped = cv.warpPerspective(image,H,(400,400))',
      'Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography. Degenerate geometry and outliers require rejection.', 'lab_03_manual:p18;lab_03_tasks:p1;lab_03_tasks:p2','sift;registration')
topic('registration','Image registration and resampling','registration;alignment;CT MRI;landmarks;correspondences',
      'Registration estimates a geometric relationship between corresponding content. Select a model appropriate to motion and scene structure, estimate it from reliable correspondences, validate alignment, then resample. Medical slices also require physical orientation and correspondence, beyond equal array dimensions.',
      'Image fusion, mosaics, change detection, and reference comparisons.', 'H, inliers = cv.findHomography(src_points,dst_points,cv.RANSAC,3.0)',
      'Different modalities may not share feature appearance. A plausible-looking overlay is not proof of accurate registration. The fusion exercise expects already registered arrays.', 'lab_02_tasks:p5;lab_03_tasks:p2','affine;perspective;sift')
topic('hog','Histogram of Oriented Gradients','HOG;HOGDescriptor;cells;blocks;gradient histogram',
      'HOG aggregates gradient orientations within cells, weights contributions by gradient strength, and normalizes groups of neighboring cells in blocks. The resulting vector describes local edge structure while reducing sensitivity to overall contrast.',
      'Handcrafted shape/texture features for a separately trained classifier.', 'from cv_core import hog_features\nfeatures = hog_features(image)',
      'A descriptor alone does not classify an object. Window, cell, block, stride and bin choices must match training. OpenCV and scikit-image configurations are not automatically interchangeable.', 'lab_04_manual:p6;lab_04_manual:p7','gradients;lbp')
topic('lbp','Local Binary Patterns','LBP;uniform LBP;local_binary_pattern;texture descriptor',
      'Compare neighboring intensities with a center pixel to form binary patterns. Uniform circular patterns have at most two bit transitions; with eight neighbors they can be summarized by ten bins including one nonuniform bin. The histogram describes local microtexture.',
      'Material texture baselines and local pattern comparisons.', 'from cv_core import uniform_lbp\nlabels, histogram = uniform_lbp(image)',
      'Noise changes threshold comparisons. Rotation properties depend on variant. The bundled replacement uses circular sampling with a cropped border and does not promise bit-for-bit equivalence with scikit-image.', 'lab_04_manual:p7;lab_04_manual:p8;lab_04_manual:p9','texture;hog')
topic('hed','Edge-direction and intensity-gradient histograms','HED;HIG;edge histogram;orientation histogram;atan2',
      'An edge-direction histogram accumulates gradient angles only at selected edge pixels. A broader gradient histogram can use all nonzero gradients and magnitude weights. Wrap signed atan2 output into 0..360 for directed angles or 0..180 for unsigned orientation before binning.',
      'Summarize dominant boundary directions and texture orientations.', 'from cv_core import direction_histogram\nhed, bins = direction_histogram(image,8,edge_only=True,signed=True,weighted=False)\nhig, bins = direction_histogram(image,9,weighted=True)',
      'The manual drops negative angles by using a positive range without wrapping and computes HED without applying its Canny mask. HED also names a different learned edge detector in other literature; this course means histogram of edge directions.', 'lab_04_manual:p9;lab_04_manual:p11;lab_04_manual:p13','gradients;histogram;hog')
topic('texture','Texture energy, contrast and classification','texture energy;local variance;local standard deviation;material classification;wood metal fabric',
      'Local intensity energy can mean the sum of squared intensities; local contrast can be measured by standard deviation. Compute means and mean squares in float, then variance=max(E[I²]-E[I]²,0). A uniformly bright patch has high intensity energy but zero contrast, so energy alone does not imply complex texture.',
      'Compare material patterns, with labeled training data and controlled sampling.', 'from cv_core import texture_stats\nenergy, contrast = texture_stats(image,7)',
      'The manual contrast code is a local sum, not the stated standard deviation, and its uint8 square can overflow. Other fields use different energy definitions such as squared filter responses: name the definition.', 'lab_04_manual:p13;lab_04_manual:p14;lab_04_manual:p15','lbp;hog;gabor')
topic('gabor','Gabor texture filters','getGaborKernel;oriented texture;frequency',
      'A Gabor filter combines a sinusoidal pattern with a Gaussian envelope to respond to a chosen orientation and spatial frequency. A bank of orientations/scales can represent textures with repeated directional structure.',
      'Supplement LBP/HOG for fabric weave or wood grain.', 'k = cv.getGaborKernel((21,21),4,np.pi/4,10,0.5,0,ktype=cv.CV_32F)\nresponse = cv.filter2D(g,cv.CV_32F,k)',
      'Scale and orientation choices matter. Responses need normalization and a classifier for material labels. More filters increase descriptor size and compute time.', 'supplement:filtering','texture;kernel')
topic('hough','Hough line and segment detection','HoughLines;HoughLinesP;Hough line transformation;rho;theta',
      'Line Hough voting accumulates evidence from edge points in parameter space. A polar representation rho=x*cos(theta)+y*sin(theta) handles vertical lines. HoughLines returns polar lines; HoughLinesP returns finite segment endpoints.',
      'Straight boundaries, lane candidates, and calibrated screen-side evidence.', 'edges = cv.Canny(g,50,150)\nlines = cv.HoughLinesP(edges,1,np.pi/180,40,minLineLength=40,maxLineGap=12)\nif lines is not None:\n    for x1,y1,x2,y2 in lines.reshape(-1,4):\n        cv.line(image,(x1,y1),(x2,y2),(0,255,0),2)',
      'A Hough line is not automatically a screen or lane. Texture can vote strongly, repeated edges cause duplicates, and lines can be absent. Handle None.', 'lab_manual_06:p17;lab_06_tasks:p1;lab_06_tasks:p2','canny;screen')
topic('screen','Screen boundaries, brightness and missing-screen hypotheses','computer lab usage;screen detection;missing screens;on off monitoring',
      'Use camera calibration/reference layout to define expected screen regions. Check Hough-supported sides and examine interior brightness. If a reference region lacks evidence, label it unconfirmed: the screen may be missing, occluded, displaced, or obscured by lighting.',
      'A constrained fixed-camera monitoring prototype.', 'from lab06 import task01_screens\nresult = task01_screens(image, [(30,60,180,150)])',
      'Dark content is not proof the power is off, and a bright reflection is not proof it is on. This solution uses calibrated front-facing rectangles; automatic perspective-screen discovery and real-data validation remain extensions.', 'lab_06_tasks:p1','hough;perspective;registration')
topic('circles','Hough circle detection','HoughCircles;coins;HOUGH_GRADIENT;minDist;param1;param2',
      'The gradient Hough circle method finds circle centers/radii from grayscale structure. dp controls accumulator resolution, minDist separates centers, param1 relates to the internal edge threshold, and param2 controls detection acceptance for the standard gradient method.',
      'Count circular candidates with a plausible radius range.', 'g = cv.medianBlur(g,5)\ncircles = cv.HoughCircles(g,cv.HOUGH_GRADIENT,1.2,30,param1=120,param2=25,minRadius=10,maxRadius=80)',
      'Parameter meanings differ for other circle methods. Perspective turns circles into ellipses. Filter duplicate detections and handle None. Avoid unsigned integer arithmetic when drawing/checking coordinates.', 'lab_manual_06:p18;lab_06_tasks:p3','median;hough')
topic('sift','SIFT keypoints and descriptors','SIFT_create;detectAndCompute;drawKeypoints;scale invariant feature transform',
      'SIFT detects stable local structures across a scale space, refines their positions, assigns orientations, and describes local gradient distributions. A standard descriptor has 128 components from 4x4 spatial cells with 8 orientation bins.',
      'Match textured planar references across moderate scale/view changes.', 'sift = cv.SIFT_create()\nkeypoints, descriptors = sift.detectAndCompute(g,None)\ndrawn = cv.drawKeypoints(image,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)',
      'Flat or repetitive surfaces give weak/ambiguous matches. descriptors can be None. SIFT invariance is approximate and does not identify identical-looking physical assets by itself.', 'lab_manual_06:p18;lab_manual_06:p20;lab_manual_06:p23','matching;ransac;orb')
topic('matching','Brute-force matching and ratio tests','BFMatcher;BruteForceMatcher;bf;knnMatch;NORM_L2;NORM_HAMMING',
      'A brute-force matcher compares descriptors. For float SIFT descriptors use L2 distance. Retrieve the two nearest candidates and retain a best match only when it is sufficiently better than the second; geometry then rejects additional outliers.',
      'Reference recognition, inventory baselines and panorama alignment.', 'bf = cv.BFMatcher(cv.NORM_L2)\npairs = bf.knnMatch(desc1,desc2,k=2)\ngood = [p[0] for p in pairs if len(p)==2 and p[0].distance < 0.75*p[1].distance]',
      'The Python constructor is cv.BFMatcher, not cv.BruteForceMatcher. Guard empty descriptors and short pairs. Binary ORB descriptors normally use Hamming distance, with Hamming2 for ORB WTA_K=3/4.', 'lab_manual_06:p21;lab_06_tasks:p1','sift;ransac;orb')
topic('ransac','RANSAC and geometric verification','findHomography;RANSAC;inliers;outlier rejection',
      'RANSAC repeatedly estimates geometry from small samples and scores consistent correspondences. A final model supported by enough inliers is more reliable than descriptor similarity alone. Check projected shape and canvas size as well as match counts.',
      'Reject false feature matches before localization/stitching.', 'H, inliers = cv.findHomography(src_points,dst_points,cv.RANSAC,4.0)\nif H is None: raise ValueError("No reliable homography")',
      'Four matches suffice algebraically but not necessarily for reliability. Collinear/repetitive points and parallax can fool a model. Inspect inlier distribution and reprojection error on real data.', 'lab_manual_06:p21;lab_manual_06:p22','sift;perspective')
topic('orb','ORB as a compact alternative','ORB_create;binary descriptors;fast keypoints',
      'ORB combines corner detection and binary descriptors to provide an efficient local feature baseline. Binary descriptors are compact and matched using bit differences, unlike SIFT float vectors.',
      'When CPU speed or descriptor storage matters more than SIFT robustness.', 'orb = cv.ORB_create(nfeatures=1000)\nkp, desc = orb.detectAndCompute(g,None)\nbf = cv.BFMatcher(cv.NORM_HAMMING)',
      'Use the correct distance norm; ORB and SIFT descriptors cannot be mixed. Accuracy depends on image content and transformation.', 'supplement:matching','sift;matching')
topic('panorama','Panorama stitching and valid-pixel masks','stitching;panorama;mosaic;blending seam',
      'Match overlapping views, verify a homography, compute the combined transformed bounds, translate them into a positive canvas, and warp both image and validity mask. Blend only where both images contribute valid pixels.',
      'The manual-point and SIFT-based panorama exercises.', 'from lab06 import task05_panorama\npanorama, evidence = task05_panorama([image1,image2])',
      'Black pixels can be valid image content, so intensity>0 is not a validity mask. Sequential stitching accumulates drift; parallax, exposure differences and moving objects need more advanced handling.', 'lab_03_tasks:p1;lab_06_tasks:p2','perspective;matching')
topic('wavelets','Haar wavelets, multiscale analysis and denoising','DWT;CWT;WPT;inverse wavelet;Haar;wavelet decomposition',
      'Wavelets separate coarse approximations from localized detail at multiple scales. A Haar pair uses (a+b)/sqrt(2) and (a-b)/sqrt(2). DWT recursively decomposes the approximation; wavelet packets also decompose details. CWT samples a family of scales/shifts more densely.',
      'Multiscale texture or the supplied one-dimensional sensor exercise.', 'from cv_core import haar_denoise\ndenoised, threshold = haar_denoise(signal,levels=3)',
      'Thresholding removes some detail and may suppress anomalies themselves. Boundary padding and scale affect output. The sensor exercise is preserved as an appendix because it is not image-specific.', 'lab_manual_06:p3;lab_manual_06:p4;lab_06_tasks:p1','anomaly;texture')
topic('anomaly','Baseline-calibrated anomaly scores','anomaly detection;sensor anomaly;MAD;residual',
      'Compare a measured statistic with a baseline distribution. The teaching wavelet example thresholds the signal minus its approximation-only Haar reconstruction using robust median absolute deviation estimated from clean baseline data. Denoising is a separate output. It returns indices and scores rather than claiming a cause.',
      'Explore transient sensor deviations or explicitly defined visual-change policies.', 'from lab06 import task03_wavelet_anomalies\nresult = task03_wavelet_anomalies(signal,clean_baseline)',
      'A single threshold cannot detect every anomaly type. Slow drift may stay in the approximation. Evaluate false alarms and missed events on labeled sequences.', 'lab_06_tasks:p1;lab_06_tasks:p2','wavelets;video')
topic('video','Offline video processing and frame loops','VideoCapture;VideoWriter;read frame;release;video offline',
      'Open a local video, check that the decoder opened, read until decoding ends, process each frame, and release the capture/writer even if an error occurs. Output frames must keep the writer size/type consistent.',
      'Echo visualization, reference recognition, lanes, or zone monitoring.', 'from cv_core import process_video\ncount = process_video("input.mp4", frame_function, "output.mp4")',
      'Codec support depends on the installed build. This writer creates silent processed video; it does not preserve audio. Camera movement invalidates a fixed background reference.', 'lab_02_tasks:p6;lab_06_tasks:p2','screen;anomaly')
topic('evaluation','Evaluate masks, matches and monitoring systems','IoU;intersection over union;precision;recall;validation;ground truth',
      'Use labeled evaluation data that represents intended conditions. For binary masks, IoU is intersection divided by union. Detection/matching needs localization and false-positive checks. Monitoring also needs sequence-level false alarms, misses, and latency.',
      'Choose among methods and tune parameters without guessing performance.', 'a, b = predicted > 0, truth > 0\nunion = np.count_nonzero(a | b)\niou = np.count_nonzero(a & b)/union if union else 1.0',
      'Synthetic tests verify controlled behavior, not real-world accuracy. Do not select thresholds on the same data used to claim final performance.', 'lab_05_tasks:p10;lab_05_tasks:p11','segmentation;ransac')
topic('calibration','Camera calibration, distortion and physical scale','calibrateCamera;undistort;intrinsics;lens distortion',
      'Camera calibration estimates focal parameters and lens distortion from multiple views of known geometry. Undistortion corrects curved projections of straight structures before geometric measurement. Physical size estimates also need distance, plane geometry, or a calibrated reference.',
      'Improve screen/document geometry and measurements when lens distortion is significant.', 'corrected = cv.undistort(image,camera_matrix,distortion_coefficients)',
      'Calibration arrays must come from actual calibration, not invented constants. Pixel width alone is not a physical measurement.', 'supplement:geometry','perspective;screen')
topic('advanced','Related CV topics: conceptual coverage and expansion boundary','CNN;deep learning;optical flow;tracking;SURF;restoration;compression;active contours;snakes;mean shift;DBSCAN;graph cuts',
      'The manuals also mention learned recognition/segmentation, motion tracking, optical flow, restoration, compression, active contours and graph/clustering methods. Their source discussions are preserved, but the current executable set focuses on the 41 supplied exercises. Learned models require separately obtained weights and representative evaluation data.',
      'Choose a later expansion when handcrafted image evidence is insufficient.', '',
      'This corpus does not contain every algorithm or answer in computer vision. Do not present conceptual mentions as implemented methods. No deep-learning weights or general language model are included.', 'lab_01_manual:p5;lab_01_manual:p11;lab_01_manual:p12;lab_05_manual:p5;lab_05_manual:p16','segmentation;evaluation')

# Supplemental related topics beyond the supplied manuals (see supplement_topics.py;
# every snippet is executed by test_supplement.py).
import supplement_topics
for _t in supplement_topics.TOPICS:
    topic(*_t)
    topics[-1]['provenance']=supplement_topics.PROVENANCE
    topics[-1]['review_status']='curated_supplement_tested'
_adv=next(t for t in topics if t['id']=='advanced')
_adv['answer']+=' Optical flow, mean-shift/CamShift tracking, background subtraction, Haar cascades, GrabCut, stereo depth and cv.dnn inference now have dedicated supplemental topic records.'
_adv['related']+=['optical_flow','meanshift','background','cascades','grabcut','stereo','dnn','fourier']

# Prerequisite setup code so every topic fragment is self-contained (see topic_setups.py).
import topic_setups
for _t in topics:
    _setup,_req=topic_setups.SETUPS.get(_t['id'],('',[]))
    _t['setup']=_setup; _t['requires']=_req
    _t['aliases']+=[a for a in topic_setups.EXTRA_ALIASES.get(_t['id'],[]) if a not in _t['aliases']]
    if _t['id'] in topic_setups.NEEDS_SOLUTIONS:
        _t['code_context']+='; needs outputs/cv_knowledge_base/solutions on sys.path'

# Explain each task, retaining its complete source pages independently in pages.jsonl.
tasks=[]
_KEEP_GLUED = re.compile(r'^(u?int\d+|float\d+|\d+D|cv2|getRotationMatrix2D|BGR2\w+|L\d+-T\d+|L[12]|(lab|task)\d.*)$')


def spaced(text):
    """Restore spaces lost between words and numbers in the task strings: 'and45-degree' -> 'and 45-degree',
    'all18combinations' -> 'all 18 combinations'. Type and API names such as uint8 or 2D stay glued."""
    def fix(m):
        word = m.group(0)
        if _KEEP_GLUED.match(word.strip('.,')):
            return word
        word = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', word)
        word = re.sub(r'(\d)x (\d)', r'\1x\2', word)          # keep sizes like 500x500 together
        return re.sub(r'(?<=\d)(?=[a-z]{3,})', ' ', word)
    return re.sub(r'[A-Za-z0-9][A-Za-z0-9.,]*', fix, text)


def task(lab,n,title,pages,entry,steps,assumptions,topics_,source=None,scope='core'):
    source=source or f'lab_{lab:02d}_tasks'
    title,steps,assumptions=spaced(title),spaced(steps),spaced(assumptions)
    tasks.append(dict(id=f'L{lab:02d}-T{n:02d}',title=title,source_refs=[f'{source}:p{p}' for p in pages],
        implementation='solutions/'+entry.split('.')[0]+'.py',entry_point=entry,solution_steps=steps.split('|'),
        assumptions_and_limits=assumptions,topic_ids=topics_.split(';'),scope=scope,
        provenance='Original worked solution derived from supplied task; statements normalized, exact original remains in PDF/page scan.',
        validation='Synthetic execution and invariant checks; course-data evaluation pending'))
task(1,1,'GroceryManager class',[1],'lab01.GroceryManager','Store item quantity and unit price in a dictionary.|Validate input; add quantities when an item is repeated.|Remove by name and raise KeyError if absent.|Return a copy of the list and sum quantity times price.','Supporting Python exercise; repeated-item behavior is an explicit implementation choice.','numpy',scope='appendix_python')
task(1,2,'Nested student records and highest average',[1],'lab01.task02_students','Compute a mean for each nonempty grade collection.|Find the highest mean and return all tied student names.|Filter records by a case-insensitive major.','Empty grades excluded; ties preserved. Supporting Python exercise.','numpy',scope='appendix_python')
task(1,3,'Load Sukuna image and format display',[2],'lab01.task03_display','Decode Sukuna.jpeg; fail clearly on missing input.|Convert BGR to RGB.|Plot at 8x6 inches with axes hidden and dark-red 16-point title with pad15.','User must supply Sukuna.jpeg; synthetic demo substitutes a labeled generated image only for testing.','io;display')
task(1,4,'Five concentric circles and bounding box',[2],'lab01.task04_circles','Create a black 800x800 color canvas.|Draw five filled alternating circles from largest to smallest.|Draw the outer bounding box and report center/bounds.','Drawing center (400,400); geometric pixel-center midpoint is (399.5,399.5). Chosen radii300,240,180,120,60.','drawing;pixels')
task(1,5,'25x25 blur and exact center ROI',[2,3],'lab01.task05_blur_roi','Check image is at least300x300.|Blur the whole original with a25x25 Gaussian.|Extract the same center300x300 coordinates from both.|Plot side by side with titles and axes hidden.','High-resolution course image missing; compare actual texture retention once supplied.','gaussian;crop')
task(1,6,'Transparent blue bottom banner',[3],'lab01.task06_overlay','Copy the image and fill the bottom20percent with blue.|Blend copy/original with opacity and its complement.|Add readable text inside the banner.','Opacity0.4 and text are configurable. BGR blue=(255,0,0).','blend;drawing')
task(1,7,'Thresholding and45-degree scaled rotation',[3],'lab01.task07_threshold_rotate','Convert to grayscale.|Compare global127 with local Gaussian threshold.|Build getRotationMatrix2D at45degrees and scale0.8.|Expand output bounds to preserve transformed content.','Scale0.8 alone does not prevent cropping on a same-size square canvas; expansion corrects that assumption.','threshold;adaptive;affine')
task(1,8,'500x500 mask-based composition',[3],'lab01.task08_composite','Resize both images to500x500.|Draw a circular binary mask.|Select foreground with the mask and background with its inverse.|Combine disjoint selections using bitwise_or.','Circle radius180 chosen; distortion from fixed square resize follows the task.','mask;resize')
task(1,9,'RGB statistics and optional Pandas describe',[3,4],'lab01.task09_rgb_statistics','Convert BGR to RGB and flatten to rows of pixels.|Calculate count, mean, sample standard deviation, min, quartiles and max.|The optional task09_optional_pandas follows the original DataFrame.describe request.','Core uses NumPy to respect requested dependency scope; optional Pandas path is preserved but not exercised.','numpy;colors')
task(2,1,'X-ray enhancement and pseudocolor',[4,5],'lab02.task01_xray','Convert to grayscale and equalize histogram.|Apply JET colormap.|Apply explicitly simulated channel gains.|Threshold high intensities.|Compare log and gamma0.6 transforms.','8-bit educational visualization; no lesion diagnosis or physical density calibration. Original images absent.','equalization;pseudocolor;gamma;threshold')
task(2,2,'CT/MRI registered-slice fusion',[5,6],'lab02.task02_fusion','Require registered corresponding slices of the same shape.|Equalize each grayscale image.|Apply distinct colormaps.|Blend CT0.7 and MRI0.3.|Show log and gamma versions.','registered=True asserts external registration has been performed; this function cannot verify anatomy automatically.','registration;blend;gamma')
task(2,3,'Echo video visualization',[6,7],'lab02.task03_echo_video','Open local video and validate decoder.|Process each frame with grayscale equalization, intensity pseudocolor, gains, log and gamma.|Concatenate original and processed views.|Save a silent video and release resources.','Colors visualize intensity; not Doppler blood flow. Dataset links preserved in source pages; no dataset downloaded.','video;equalization;pseudocolor')
task(3,1,'Manual matrix fingerprint enlargement',[3],'lab03.task01_scale','Construct a diagonal2D scale matrix and embed in3x3 homogeneous form.|Translate source center to origin and destination center into place.|Warp to three-times width and height.','Task says enlarge by300percent: default interprets as to300percent (factor3); use4 for a literal300percent increase.','coordinates;affine;resize')
task(3,2,'Undo satellite rotation with sin/cos',[3],'lab03.task02_undo_rotation','Build a2x2 inverse rotation from sine/cosine.|Center the rotation using homogeneous translations.|Transform corners, compute bounds, and warp to an expanded canvas.','Default corrects a45degree clockwise image rotation. Direction must match actual input.','coordinates;affine')
task(3,3,'Barcode horizontal de-shearing',[3],'lab03.task03_deshear','Represent observed horizontal shear as xprime=x+k*y.|Use inverse coefficient minus k.|Warp to expanded output bounds.','Shear coefficient is not supplied by task; default0.3 is illustrative and configurable.','affine;coordinates')
task(3,4,'Map translation recovery',[3],'lab03.task04_translate','Undo150pixels left and80pixels up using translation(+150,+80).|Create an output canvas that accommodates the positive shift.|Warp with the3x3 matrix.','Translation cannot restore image content that was already cropped away in the input.','coordinates;affine')
task(3,5,'Rigid transformation',[2],'lab03.task05_rigid','Build rotation matrix.|Compose translation after rotation.|Apply once to the image and retain matrix for point checks.','Parameters are examples because the exercise provides no numeric correspondences. Canvas may need adjustment.','affine;coordinates')
task(3,6,'Similarity transformation',[2],'lab03.task06_similarity','Compose uniform scale, rotation, then translation.|Apply a single warp.|Verify all distances scale by the same positive factor.','Default values illustrative. Similarity preserves angles but not absolute lengths.','affine;coordinates')
task(3,7,'Six-unknown affine solve',[2],'lab03.task07_affine','Build six linear equations from three point pairs.|Reject collinear source landmarks.|Solve the six coefficients with np.linalg.solve.|Warp into the destination frame.','Requires real corresponding landmarks; synthetic landmarks only validate the solver.','affine;registration')
task(3,8,'Top-down square perspective rectification',[2],'lab03.task08_rectify','Provide four ordered convex corners.|Associate them with square corners.|Estimate getPerspectiveTransform and warpPerspective.','Corner order TL,TR,BR,BL; target400x400 default may distort a nonsquare physical object.','perspective')
task(3,9,'Panorama from field correspondences',[1],'lab03.task09_panorama','Provide at least four paired landmarks.|Estimate right-to-left homography and reject failure.|Compute full canvas bounds.|Warp validity masks and average overlapping valid content.','Planar scene or approximately pure camera rotation assumed; parallax and exposure seams remain limitations.','perspective;panorama;registration')
task(3,10,'Painting insertion using transform composition',[1],'lab03.task10_painting','Build linear scaling and rigid rotation/translation matrices.|Transform painting corners through those stages.|Estimate projective mapping from those intermediate corners to the wall quadrilateral.|Compose all matrices and warp original once.|Insert using a warped all-ones mask.','Black painting pixels remain valid. Specifying the final four corners determines the final mapping; intermediate operations illustrate composition.','coordinates;perspective;mask')
task(4,1,'Material texture analysis with HOG and LBP',[15],'lab04.task01_features','Compute circular uniform LBP histogram for local microtexture.|Compute normalized block HOG for directional structure.|Compare optional local energy and contrast.|For an actual label, compare with labeled exemplars via classify_material.','LBP: useful microtexture, weak under noise/scale change. HOG: useful grain/weave direction, weak under rotation and texture ambiguity. No wood/metal/fabric accuracy claimed without a labeled dataset.','hog;lbp;texture',source='lab_04_manual')
task(5,1,'Global versus adaptive document segmentation',[1,2],'lab05.task01_document','Apply global inverse thresholds80,127,180.|Apply adaptive Gaussian inverse threshold.|Compare masks and illumination failures; with known truth use task10_compare.','Threshold polarity is dark text as foreground. Which method is best depends on the supplied document.','threshold;adaptive;evaluation')
task(5,2,'Adaptive parameter sweep',[2,3],'lab05.task02_adaptive_sweep','Try mean and Gaussian local thresholds.|Cross block sizes11,31,51 with C2,7,12.|Display all18combinations and inspect broken/merged strokes.','BINARY output: increasing C generally makes more white. No universal optimal block size.','adaptive;evaluation')
task(5,3,'Otsu coin segmentation and histogram changes',[3],'lab05.task03_otsu','Convert to grayscale and record histogram.|Run Otsu and retain its numeric threshold.|Repeat after5x5 blur and a documented contrast change.|Compare three masks and histograms.','Contrast transform alpha 1.2 beta10 is an experiment; clips highlights. Histograms are returned and plotted by run_synthetic.py.','otsu;histogram')
task(5,4,'HSV yellow-car isolation',[4],'lab05.task04_yellow','Convert BGR to HSV.|Create a narrow yellow mask and broader yellow mask.|Extract pixels with each mask.|Inspect background leakage versus missed dim yellow areas.','Fixed ranges are starting examples, not calibrated for missing car images. Hue is0..179 for uint8.','colors;mask')
task(5,5,'Canny high-threshold comparison',[5],'lab05.task05_canny','Convert and Gaussian smooth.|Hold low threshold30 constant and try high60,120,200.|Compare connected weak edges and suppression.','Final output cannot identify original strong/weak classes; that would require access to intermediate gradients/NMS.','canny')
task(5,6,'Two seeds and three region-growth tolerances',[6],'lab05.task06_region_grow','Provide two seed points.|Grow with fixed-seed tolerances5,15,30.|Compare six masks for leakage and incomplete coverage.','Course MRI and seeds absent. This is intensity segmentation, not a clinical lesion classifier.','region;evaluation')
task(5,7,'Complete marker-based watershed pipeline',[7],'lab05.task07_watershed','Convert to grayscale and choose foreground polarity.|Otsu threshold; open the mask.|Dilate to background support.|Distance transform and threshold for foreground.|Subtract to find unknown.|Label seeds; add1 so known background=1; unknown=0.|Run watershed on BGR uint8 with int32 markers.|Mark boundaries red and show all stages.','Red is BGR(0,0,255). Empty/poor markers can leave objects unseparated.','watershed;distance;morphology;components')
task(5,8,'Distance-threshold effects on watershed',[8],'lab05.task08_distance_sweep','Repeat full watershed for fractions0.2,0.4,0.6,0.8 of maximum distance.|Record foreground marker count and final labeled-region count.|Compare overlays and explain merged/lost seeds.','A higher fraction is not always better; small objects can lose all seeds. Counts depend on image.','distance;watershed')
task(5,9,'K=2,4,6 image clustering',[9],'lab05.task09_kmeans','Reshape BGR pixels to float32 rows.|Run K-means for2,4,6 with recorded seed and stopping criteria.|Reconstruct center-colored images.|Compare compactness, color detail and false merges.','Spatial position is not included in this baseline; disconnected areas of the same color share a label.','kmeans;numpy')
task(5,10,'Compare three segmentation methods',[10,11],'lab05.task10_compare','Choose a dark-text document as the common task.|Compare fixed127, Otsu and adaptive Gaussian masks.|Compute IoU only when ground truth is supplied.|Explain global failures under illumination changes and adaptive block/C tradeoffs.','Includes measured synthetic results in validation.json; original-data comparison remains pending. No method wins universally.','evaluation;threshold;otsu;adaptive')
task(6,1,'Hough-supported screen monitoring',[1],'lab06.task01_screens','Read image, grayscale, smooth and Canny.|Detect Hough line segments and draw them on a black canvas.|Check support around expected screen rectangles.|Measure interior brightness.|Display supported boundaries and unconfirmed locations.','Requires calibrated front-facing ROIs. Outputs visible_dark/visible_bright/unconfirmed; absence may be missing or occluded, brightness is not proven power state.','screen;hough;canny')
task(6,2,'SIFT asset inventory',[1],'lab06.task02_inventory','Provide named reference pictures for assets.|Compute SIFT descriptors and match with BF L2 ratio test.|Validate planar localization by RANSAC.|Return evidence for each reference and explicit failure reasons.','Identical-looking assets cannot be uniquely identified by appearance alone. Multiple instances and untextured screens require extensions.','sift;matching;ransac')
task(6,3,'Wavelet sensor anomalies',[1,2],'lab06.task03_wavelet_anomalies','Decompose clean baseline with Haar DWT.|Estimate noise from detail coefficients and soft-threshold for a denoised output.|Separately reconstruct approximation-only signals and compute detail residuals.|Calibrate a cutoff using clean-baseline residual MAD.|Apply the same decomposition to the test signal and return deviations.','Signal-processing appendix preserved from course. Baseline assumed representative; broad/slow anomalies can be missed. Soft-threshold denoising residuals are not used as anomaly scores because they can cap retained impulses.','wavelets;anomaly',scope='appendix_signal')
task(6,4,'Reference object recognition in images/video',[2],'lab06.task04_recognize','Extract SIFT reference and scene features.|Apply ratio test, RANSAC and quadrilateral plausibility checks.|Draw localized boundary when evidence passes.|Use task04_video for a local video loop.','Requires textured planar reference; returns failure if descriptors/matches are insufficient.','sift;matching;video')
task(6,5,'Automatic panorama from overlapping images',[2],'lab06.task05_panorama','Use SIFT to align each next image to the current panorama.|Reject weak homographies.|Expand canvas and blend valid overlap.|Record inlier/match counts per step.','Sequential baseline can drift; no bundle adjustment, exposure compensation or parallax handling.','panorama;sift;ransac')
task(6,6,'Hough lane-line detection',[2,3],'lab06.task06_lanes','Smooth and detect edges.|Restrict to a road-shaped trapezoid.|Detect segments and group by slope/side.|Fit x as a function of y and draw left/right boundaries.','Fixed-camera straight-lane teaching example, not a vehicle-control system. ROI/slope assumptions fail on curves/hills.','hough;canny')
task(6,7,'Coin counting with Hough circles',[3],'lab06.task07_coins','Grayscale and median blur.|Run HoughCircles with plausible radius bounds and center separation.|Safely handle no detections.|Draw centers/circumferences and count candidates.','Perspective, touching coins and reflections change performance; tune on labeled actual images.','circles;median')
task(6,8,'Restricted-zone visual-change monitor',[3],'lab06.ZoneMonitor','Provide empty background and zone polygon.|Compute absolute frame/background difference.|Threshold and open the mask; restrict to zone.|Filter small components.|Require consecutive frames and emit one local event per active episode.','Explicit policy is any new foreground in the zone; visual change does not prove a person/object is unauthorized. Requires static camera. No external alarm is sent.','video;mask;morphology;anomaly')

corrections=[]
def correction(id,refs,issue,correct):
    corrections.append({'id':id,'source_refs':refs.split(';'),'issue':issue,'correction':correct,'status':'curated technical correction; original source preserved'})
correction('C01','lab_01_manual:p10','Coordinate diagram can confuse row/column with x/y.','Use image[y,x], with x column, y row; OpenCV points use (x,y).')
correction('C02','lab_01_manual:p14','Prose describes cv.imshow/waitKey but displayed code uses Matplotlib.','The reconstructed example saves a Matplotlib figure; GUI calls are a separate alternative.')
correction('C03','lab_01_manual:p17','Comment describes5x5 blur while code uses31x31.','Record actual example as31x31; Task05 independently requests25x25.')
correction('C04','lab_01_manual:p19','Text scale in code/prose differs.','Reconstructed example uses a documented visible scale1.5; preserve original scan for exact original constant.')
correction('C05','lab_01_manual:p21','Threshold maxval200 is described as pure white.','uint8 white is255;200 is a lighter gray although auto-scaled plots may show it white.')
correction('C06','lab_01_manual:p23','cv.add is presented as image blending.','cv.add is saturating addition; cv.addWeighted with complementary weights implements alpha blending.')
correction('C07','lab_01_tasks:p3','Scale0.8 at45degrees assumed to prevent cropping.','For a square,0.8*sqrt(2)>1; expand canvas or use a smaller scale if preserving all content.')
correction('C08','lab_03_tasks:p3','Enlarge by300percent is ambiguous.','Default interprets final size300percent (3x); strict increase by300percent means4x. Both are documented.')
correction('C09','lab_03_tasks:p1;lab_03_tasks:p2;lab_03_tasks:p3','Physical PDF page order reverses exercise groups.','Canonical task order1..10 maps to physical pages3,2,1. All references use physical PDF pages.')
correction('C10','lab_03_manual:p17','Homogeneous notation motivates sweeping performance claims about vector addition/GPU cost.','The reliable benefit here is unified representation and composition; speed depends on implementation and hardware.')
correction('C11','lab_04_manual:p11;lab_04_manual:p13','Signed atan2 angles are histogrammed over positive-only ranges.','Wrap angles modulo360 or180 before binning; otherwise negative angles are discarded.')
correction('C12','lab_04_manual:p11','HED creates a Canny map but histograms all gradient orientations.','Restrict to selected edge pixels for an edge-direction histogram, and document weighting.')
correction('C13','lab_04_manual:p13','Histogram normalization implies scale invariance.','Normalization reduces dependence on sample count/overall weighting; it does not generally make texture spatially scale invariant.')
correction('C14','lab_04_manual:p13;lab_04_manual:p14','High sum-of-squared intensity energy is equated to complex texture.','Uniform white has high intensity energy and zero contrast; distinguish brightness energy from variation.')
correction('C15','lab_04_manual:p14;lab_04_manual:p15','uint8 squared intensities overflow, output may saturate, and contrast code sums pixels.','Convert to float before squaring; local standard deviation=sqrt(max(E[I²]-E[I]²,0)).')
correction('C16','lab_04_manual:p17','A getGaussianKernel column is used as if a full2D blur.','Use outer product v@v.T or sepFilter2D with both directions.')
correction('C17','lab_04_manual:p18;lab_04_manual:p19','filter2D same-size output is labeled full/valid; reflected borders called zero padding.','Explicitly pad for full, crop for valid, use BORDER_CONSTANT for zero padding; same describes shape, not border rule.')
correction('C18','lab_04_manual:p19','filter2D is passed strides=(2,2), an unsupported argument.','Filter then subsample explicit valid output, or use another operation that actually exposes stride.')
correction('C19','lab_04_manual:p16;lab_04_manual:p18','filter2D called convolution without the correlation distinction.','Flip an asymmetric kernel in both axes for true convolution.')
correction('C20','lab_04_manual:p21','Sobel-Feldman described as a separate directional variation.','Sobel-Feldman is the Sobel operator name; derivative directions are selected explicitly.')
correction('C21','lab_manual_06:p11','Non-maximum suppression described only as eliminating weak pixels.','NMS keeps local maxima along gradient direction; hysteresis thresholds then select connected edges.')
correction('C22','lab_05_manual:p11','Region-growth coordinate names differ from OpenCV point convention.','Bundled API consistently accepts(x,y), indexes[y,x], checks seed, and marks queued coordinates visited.')
correction('C23','lab_05_manual:p13','Boundary [255,0,0] called red in a BGR array; Matplotlib display is unconverted.','BGR red=(0,0,255), followed by BGR2RGB when plotting. Display conversion and array convention must agree.')
correction('C24','lab_05_manual:p15','K-means BGR array sent directly to Matplotlib.','Convert BGR to RGB for plotting; retain BGR for subsequent OpenCV operations.')
correction('C25','lab_manual_06:p15;lab_manual_06:p16','Absolute LoG image can be mistaken for zero-crossing edges.','Preserve signed response and explicitly detect sign changes for zero-crossing localization.')
correction('C26','lab_manual_06:p17','Slope/intercept line description omits vertical-line limitation.','OpenCV polar Hough uses rho/theta, covering vertical lines without infinite slope.')
correction('C27','lab_manual_06:p21','Illustration small text says120-dimensional while grid is4x4x8.','Standard SIFT descriptor is128-dimensional.')
correction('C28','lab_manual_06:p23','descriptors.shape accessed unconditionally.','No keypoints may return descriptors=None; guard before shape/matching.')
correction('C29','lab_06_tasks:p1','Line boundaries/brightness alone treated as enough for screen missing/on-off status.','Require expected layout and report visual evidence; missing vs occluded and dark vs powered-off are distinct.')
correction('C30','lab_02_tasks:p6','Intensity colorization can be read as blood-flow information.','JET of grayscale carries only intensity; it is not measured Doppler velocity.')
correction('C31','lab_02_tasks:p5','Fusion can be attempted without explicit registration.','Require already registered corresponding slices; resizing alone is insufficient.')
correction('C32','lab_manual_06:p1','Lab Manual06 filename has a cover labeled Lab05.','Keep the supplied filename identity lab_manual_06; map its wavelet/edge/Hough/SIFT content to the supplied Lab06 tasks, and preserve the cover discrepancy.')

# Public references: links + concise authored relevance, never copied manuals.
external=[
 ('filtering','OpenCV filtering API','https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html','Correlation, border rules and filter signatures.'),
 ('smoothing','OpenCV smoothing tutorial','https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html','Box, Gaussian, median and bilateral choices.'),
 ('gradients','OpenCV gradients','https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html','Signed derivative depth and gradient display.'),
 ('border','OpenCV borders','https://docs.opencv.org/4.x/dc/dc3/tutorial_copyMakeBorder.html','Padding boundary behavior.'),
 ('threshold','OpenCV thresholding','https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html','Global, adaptive and Otsu interfaces.'),
 ('morphology','OpenCV morphology','https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html','Structuring-element operations.'),
 ('watershed','OpenCV watershed','https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html','Markers, unknown regions and watershed boundary labels.'),
 ('matching','OpenCV feature matching','https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html','Matcher norms and nearest-neighbor matching.'),
 ('homography','OpenCV feature homography','https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html','Geometric match filtering.'),
 ('contours','OpenCV contour features','https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html','Polygon and shape measurements.'),
 ('equalize','OpenCV equalization','https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html','Equalization and CLAHE.'),
 ('kmeans','OpenCV K-means','https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html','Float samples, criteria, centers and labels.'),
 ('circles','OpenCV Hough circles','https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html','Circle parameterization.'),
 ('geometry','OpenCV geometric transformations','https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html','Affine and projective image warping.'),
 ('imshow','Matplotlib imshow','https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html','Image array interpretation and display range.'),
 ('histogram','NumPy histogram','https://numpy.org/doc/stable/reference/generated/numpy.histogram.html','Bin edges, range and weighted counting.'),
 ('lbp','scikit-image feature reference','https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern','Reference for manual HOG/LBP examples; runtime replacement differences documented.')]
external+=supplement_topics.EXTERNAL
external=[dict(id='supplement:'+x[0],title=x[1],url=x[2],relevance=x[3],accessed='2026-10-07',type='primary_documentation',stored_fulltext=False) for x in external]

def write_jsonl(name,records):
    (ROOT/'records'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
# Atomic manual-grounded facts for exam-style questions (see facts_manuals.py).
import facts_manuals
_topic_ids={t['id'] for t in topics}
facts=[]
for _n,(_ref,_topic,_stmt,_note) in enumerate(facts_manuals.FACTS,1):
    facts.append(dict(id=f'fact:{_n:04d}',statement=_stmt,note=_note,topic_id=_topic,source_refs=[_ref],
        provenance=('Read from the installed library' if _ref.startswith('opencv_introspection') else 'Restated from OpenCV documentation or CV fundamentals' if _ref.split(':')[0] in ('opencv_docs','cv_fundamentals') else 'Restated from the cited manual page; note gives the technically precise view where the manual is imprecise.'),
        scope='core',review_status='curated_fact'))
write_jsonl('facts.jsonl',facts)
import facts_opencv_api
triples=[]
for _n,(_subj,_attr,_val,_ref) in enumerate(facts_opencv_api.TRIPLES+__import__('facts_gaps').TRIPLES+facts_manuals.INTRO_TRIPLES,1):
    triples.append(dict(id=f'triple:{_n:04d}',subjects=_subj,attribute=_attr,value=_val,source_refs=[_ref],scope='core'))
write_jsonl('triples.jsonl',triples)
write_jsonl('topics.jsonl',topics); write_jsonl('tasks.jsonl',tasks); write_jsonl('corrections.jsonl',corrections)
(ROOT/'records'/'external_sources.json').write_text(json.dumps(external,indent=2),encoding='utf-8')
(ROOT/'records'/'terminology.json').write_text(json.dumps({
    'cv2.BruteForceMatcher':{'canonical':'cv2.BFMatcher','status':'corrected API name'},
    'SCAR':{'canonical':'Scharr','status':'supported by supplied manual topic'},
    'VGA to RGB':{'canonical':'BGR to RGB','status':'contextual correction; VGA is not a color space'},
    'botch filter':{'canonical_candidate':'box filter','status':'uncertain; Box Blur exists in manual, original term retained'},
    'R10':{'canonical_candidate':'ROI','status':'unresolved; do not automatically replace'},
    'plot dot bar':{'canonical':'matplotlib.pyplot.bar','status':'spoken API normalization'}},indent=2),encoding='utf-8')

task_md=['# Worked task solutions','', 'All page references below are **physical PDF pages**, starting at1. The original documents, full-page scans and raw OCR remain under sources/ and extracted/. These explanations and Python implementations are newly authored, corrected teaching solutions. Synthetic execution is not validation on the missing original datasets.','',
 'Run all demonstrations: `python solutions/run_synthetic.py`. For your own images, import a task function from its module; the function returns arrays/results. Use `cv_core.grid` to save image dictionaries. No internet call is used.','']
for t in tasks:
    task_md += [f"## {t['id']} — {t['title']}",'',f"Source: {', '.join(t['source_refs'])}.",'',f"Implementation: `{t['entry_point']}` in [{t['implementation']}]({t['implementation']}).",'']
    task_md += [f'{i}. {s}' for i,s in enumerate(t['solution_steps'],1)]
    task_md += ['', '**Assumptions and interpretation:** '+t['assumptions_and_limits'],'']
(ROOT/'TASK_SOLUTIONS.md').write_text('\n'.join(task_md),encoding='utf-8')
handbook=['# Computer vision knowledge handbook','', 'Scope: supplied labs plus related OpenCV/NumPy/Matplotlib fundamentals. This is a curated offline corpus, not a universal computer-vision model. Source references distinguish course grounding from additional documentation. Code fragments assume imports and input context stated in records/topics.jsonl.','']
for t in topics:
    handbook += [f"## {t['title']}",'',t['answer'],'','**Use when:** '+t['when_to_use'],'','**Pitfalls:** '+t['pitfalls'],'']
    if t.get('setup'): handbook += ['Setup (earlier steps this example assumes):','','```python',t['setup'],'```','']
    if t['code']: handbook += ['```python',t['code'],'```','']
    handbook += ['Sources: '+', '.join(t['source_refs'])+'.','']
(ROOT/'HANDBOOK.md').write_text('\n'.join(handbook),encoding='utf-8')
(ROOT/'reports'/'CORRECTIONS.md').write_text('# Corrections and ambiguities\n\nThe original scans are unchanged. These are curated corrections, not edits to course source text.\n\n'+'\n\n'.join(f"## {c['id']} ({', '.join(c['source_refs'])})\n\n{c['issue']}\n\n**Corrected interpretation:** {c['correction']}" for c in corrections),encoding='utf-8')
print(json.dumps({'topics':len(topics),'tasks':len(tasks),'corrections':len(corrections),'references':len(external)}))
