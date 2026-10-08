# Answer sheet: mcq_features_edges.txt

Graded 125/125 correct.

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. Feature extraction transforms raw visual data into:**

- A) Larger raw data
- B) More compact and meaningful representation
- C) Audio signals
- D) Only color channels

Answer: **B) More compact and meaningful representation** ✅
Evidence (nli): Feature extraction selects and transforms relevant information from raw visual data (images or videos) into a more compact and meaningful representation. — `fact:0001 lab_04_manual:p4`

**2. Raw visual data is often too ___ for direct analysis.**

- A) Simple
- B) High-dimensional and complex
- C) Low-dimensional
- D) Binary

Answer: **B) High-dimensional and complex** ✅
Evidence (nli): Raw visual data is often too complex and high-dimensional for direct analysis and interpretation by algorithms. — `fact:0004 lab_04_manual:p4`

**3. Features should capture:**

- A) All irrelevant details
- B) Essential information about objects/patterns
- C) Only noise
- D) Only pixel coordinates

Answer: **B) Essential information about objects/patterns** ✅
Evidence (nli): Features are distinctive characteristics that should capture essential information about objects, patterns or structures while discarding irrelevant details. — `fact:0006 lab_04_manual:p4`

**4. Local features are extracted from:**

- A) Entire image
- B) Specific regions such as keypoints, corners, patches
- C) Only color histograms
- D) Only edges

Answer: **B) Specific regions such as keypoints, corners, patches** ✅
Evidence (nli): Local features are extracted from specific regions of an image such as keypoints, corners or small patches. — `fact:0007 lab_04_manual:p5`

**5. Global features are computed:**

- A) Only on corners
- B) Over the entire image
- C) Only on patches
- D) Only on keypoints

Answer: **B) Over the entire image** ✅
Evidence (nli): Global features are computed over the entire image and capture its overall characteristics. — `fact:0009 lab_04_manual:p5`

**6. Which is an example of global feature?**

- A) Keypoint
- B) Corner
- C) Color histogram
- D) Small patch

Answer: **C) Color histogram** ✅
Evidence (nli): Histogram of Color (a color histogram) is an example of a global feature because it is computed over the entire image. — `fact:0047 lab_04_manual:p5`

**7. Histogram-based methods capture:**

- A) Only shape
- B) Statistical information like color or texture distributions
- C) Only depth
- D) Only motion

Answer: **B) Statistical information like color or texture distributions** ✅
Evidence (nli): Histogram-based methods: histograms of color or texture distributions capture statistical information about the image. — `fact:0011 lab_04_manual:p5`

**8. Gabor filters and Haar wavelets are examples of:**

- A) Histogram methods
- B) Filter-based methods
- C) Deep learning methods
- D) Color spaces

Answer: **B) Filter-based methods** ✅
Evidence (nli): Filter-based methods: filters like Gabor filters or Haar wavelets identify edges, textures or specific patterns. — `fact:0012 lab_04_manual:p5`

**9. CNNs extract features:**

- A) Manually
- B) Directly from data through layers
- C) Only by histograms
- D) Only by Sobel

Answer: **B) Directly from data through layers** ✅
Evidence (nli): Deep learning features: convolutional neural networks (CNNs) learn features directly from the data through the network layers. — `fact:0013 lab_04_manual:p5`

**10. Which is an application of feature extraction?**

- A) Face recognition
- B) File compression only
- C) Audio recording
- D) Printing

Answer: **A) Face recognition** ✅
Evidence (nli): Face recognition extracts facial features like eyes, nose and mouth for identity verification. — `fact:0015 lab_04_manual:p5`

**11. Feature extraction challenges include:**

- A) Lighting, scale, orientation, noise
- B) Only color
- C) Only resolution
- D) No challenges

Answer: **A) Lighting, scale, orientation, noise** ✅
Evidence (nli): Feature extraction challenges are variations in lighting, scale, orientation and noise in real-world data. — `fact:0016 lab_04_manual:p5`

**12. Choosing appropriate feature extraction techniques and parameters is:**

- A) Not important
- B) Critical for successful computer vision applications
- C) Only for audio
- D) Only for video games

Answer: **B) Critical for successful computer vision applications** ✅
Evidence (nli): Choosing appropriate feature extraction techniques and parameters is critical for successful computer vision applications. — `fact:0017 lab_04_manual:p6`

**13. Extracted features are often used with:**

- A) Matching or learning algorithms
- B) Only printers
- C) Only monitors
- D) Only keyboards

Answer: **A) Matching or learning algorithms** ✅
Evidence (nli): Extracted features are often used with matching or learning algorithms for object recognition or image classification. — `fact:0018 lab_04_manual:p6`

**14. Local features are often used for:**

- A) Image matching and object detection
- B) Color printing
- C) Audio synthesis
- D) File storage

Answer: **A) Image matching and object detection** ✅
Evidence (nli): Local features are often used for image matching and object detection. — `fact:0008 lab_04_manual:p5`

**15. Image moments are an example of:**

- A) Local feature
- B) Global feature
- C) Edge detector
- D) Filter

Answer: **B) Global feature** ✅
Evidence (nli): Examples of global features are color histograms, texture descriptors and image moments. — `fact:0010 lab_04_manual:p5`

**16. HOG stands for:**

- A) Histogram of Optical Gradients
- B) Histogram of Oriented Gradients
- C) High Order Gradients
- D) Histogram of Oriented Geometry

Answer: **B) Histogram of Oriented Gradients** ✅
Evidence (nli): HOG stands for Histogram of Oriented Gradients. — `fact:0019 lab_04_manual:p6`

**17. HOG is particularly useful for:**

- A) Audio detection
- B) Object detection and pedestrian detection
- C) Color printing
- D) File compression

Answer: **B) Object detection and pedestrian detection** ✅
Evidence (nli): HOG is particularly useful for object detection and pedestrian detection. — `fact:0020 lab_04_manual:p6`

**18. HOG captures information about:**

- A) Local gradient or edge patterns
- B) Only color
- C) Only depth
- D) Only motion

Answer: **A) Local gradient or edge patterns** ✅
Evidence (nli): HOG captures information about local gradient or edge patterns, the distribution of gradient directions in an image. — `fact:0021 lab_04_manual:p6`

**19. In HOG, image is divided into:**

- A) Small cells
- B) Large blocks only
- C) Audio frames
- D) Color bins

Answer: **A) Small cells** ✅
Evidence (nli): HOG divides the image into small cells and computes gradient magnitude and orientation for each pixel in the cells. — `fact:0022 lab_04_manual:p6`

**20. For HOG, gradient computation typically uses:**

- A) Sobel operator
- B) Fourier transform
- C) K-means
- D) PCA

Answer: **A) Sobel operator** ✅
Evidence (nli): HOG gradient computation typically uses the Sobel operator. — `fact:0024 lab_04_manual:p6`

**21. HOG first converts input image to:**

- A) RGB
- B) Grayscale
- C) HSV
- D) Binary only

Answer: **B) Grayscale** ✅
Evidence (nli): HOG step 1 image preprocessing: convert the input image to grayscale to simplify gradient calculation. — `fact:0023 lab_04_manual:p6`

**22. In HOG, histogram bins represent:**

- A) Color ranges
- B) Different orientation ranges
- C) Pixel coordinates
- D) Depth values

Answer: **B) Different orientation ranges** ✅
Evidence (nli): In HOG each cell gets a histogram of gradient orientations; the histogram bins represent different orientation ranges. — `fact:0026 lab_04_manual:p6`

**23. Block normalization in HOG reduces effects of:**

- A) Lighting variations
- B) Motion
- C) Audio noise
- D) Compression

Answer: **A) Lighting variations** ✅
Evidence (nli): HOG block normalization groups neighboring cells into blocks (typical block 2x2 cells) and normalizes them to reduce the effects of lighting variations. — `fact:0027 lab_04_manual:p7`

**24. In HOG, neighboring cells are grouped into:**

- A) Bins
- B) Blocks
- C) Layers
- D) Channels

Answer: **B) Blocks** ✅
Evidence (nli): HOG block normalization groups neighboring cells into blocks (typical block 2x2 cells) and normalizes them to reduce the effects of lighting variations. — `fact:0027 lab_04_manual:p7`

**25. Final HOG feature vector is formed by:**

- A) Averaging all pixels
- B) Concatenating normalized histograms from blocks
- C) Sorting colors
- D) Removing edges

Answer: **B) Concatenating normalized histograms from blocks** ✅
Evidence (nli): The final HOG feature vector is formed by concatenating the normalized histograms from the blocks. — `fact:0028 lab_04_manual:p7`

**26. Typical cell size example in HOG manual:**

- A) 8x8 pixels
- B) 16x16 pixels
- C) 32x32 pixels
- D) 64x64 pixels

Answer: **A) 8x8 pixels** ✅
Evidence (nli): In skimage hog, pixels_per_cell=(8,8) sets the cell size and cells_per_block=(2,2) sets the number of cells per block; orientations sets the number of bins. — `fact:0030 lab_04_manual:p7`

**27. Typical block example in HOG manual:**

- A) 2x2 cells
- B) 3x3 cells
- C) 4x4 cells
- D) 5x5 cells

Answer: **A) 2x2 cells** ✅
Evidence (nli): In skimage hog, pixels_per_cell=(8,8) sets the cell size and cells_per_block=(2,2) sets the number of cells per block; orientations sets the number of bins. — `fact:0030 lab_04_manual:p7`

**28. HOG code uses function:**

- A) cv2.hog
- B) skimage.feature.hog
- C) cv2.HOGDescriptor
- D) numpy.hog

Answer: **B) skimage.feature.hog** ✅
Evidence (nli): The HOG code uses skimage.feature.hog (from skimage.feature import hog), not an OpenCV function. — `fact:0029 lab_04_manual:p7`

**29. In HOG code, pixels_per_cell=(8,8) means:**

- A) Block size
- B) Cell size
- C) Image size
- D) Bin count

Answer: **B) Cell size** ✅
Evidence (nli): In skimage hog, pixels_per_cell=(8,8) sets the cell size and cells_per_block=(2,2) sets the number of cells per block; orientations sets the number of bins. — `fact:0030 lab_04_manual:p7`

**30. In HOG code, cells_per_block=(2,2) means:**

- A) Cell size
- B) Number of cells per block
- C) Number of orientations
- D) Number of channels

Answer: **B) Number of cells per block** ✅
Evidence (nli): In skimage hog, pixels_per_cell=(8,8) sets the cell size and cells_per_block=(2,2) sets the number of cells per block; orientations sets the number of bins. — `fact:0030 lab_04_manual:p7`

**31. LBP stands for:**

- A) Local Binary Pixel
- B) Local Binary Pattern
- C) Linear Binary Pattern
- D) Local Basic Pattern

Answer: **B) Local Binary Pattern** ✅
Evidence (nli): LBP stands for Local Binary Pattern, a texture feature extraction technique. — `fact:0032 lab_04_manual:p7`

**32. LBP is commonly used for:**

- A) Texture classification and face recognition
- B) Audio compression
- C) Motion detection only
- D) Color printing

Answer: **A) Texture classification and face recognition** ✅
Evidence (nli): LBP is particularly useful for texture classification and face recognition. — `fact:0034 lab_04_manual:p7`

**33. LBP operates on:**

- A) RGB images only
- B) Grayscale images
- C) Depth images only
- D) Audio signals

Answer: **B) Grayscale images** ✅
Evidence (nli): LBP operates on grayscale images and defines a circular neighborhood around each pixel. — `fact:0035 lab_04_manual:p8`

**34. In LBP, each pixel is compared with:**

- A) Its neighboring pixels
- B) Whole image
- C) Color histogram
- D) Sobel kernel

Answer: **A) Its neighboring pixels** ✅
Evidence (nli): LBP captures local patterns by comparing the intensity of a pixel with its neighboring pixels. — `fact:0033 lab_04_manual:p7`

**35. In LBP, if neighbor intensity is greater or equal to center, assign:**

- A) 0
- B) 1
- C) -1
- D) 255

Answer: **B) 1** ✅
Evidence (nli): In LBP, if the neighbor intensity is greater than or equal to the center pixel intensity, the neighbor is assigned 1. — `fact:0036 lab_04_manual:p8`

**36. In LBP, if neighbor intensity is less than center, assign:**

- A) 0
- B) 1
- C) -1
- D) 255

Answer: **A) 0** ✅
Evidence (nli): In LBP, if the neighbor intensity is less than the center pixel intensity, the neighbor is assigned 0. — `fact:0037 lab_04_manual:p8`

**37. LBP binary values are concatenated to form:**

- A) Color histogram
- B) LBP pattern
- C) Sobel gradient
- D) HOG block

Answer: **B) LBP pattern** ✅
Evidence (nli): LBP concatenates the binary comparison values to form an LBP pattern, a binary number. — `fact:0038 lab_04_manual:p8`

**38. The LBP histogram serves as:**

- A) Feature vector
- B) Kernel
- C) Threshold
- D) Edge map

Answer: **A) Feature vector** ✅
Evidence (nli): The HED feature vector is the histogram of edge directions, counting pixels in each orientation bin. — `fact:0055 lab_04_manual:p10`

**39. LBP code uses function:**

- A) cv2.LBP
- B) skimage.feature.local_binary_pattern
- C) np.lbp
- D) cv2.local_binary_pattern

Answer: **B) skimage.feature.local_binary_pattern** ✅
Evidence (nli): The LBP code uses skimage.feature.local_binary_pattern (from skimage import feature). — `fact:0041 lab_04_manual:p8`

**40. In LBP code, method='uniform' is used to:**

- A) Reduce number of patterns / use uniform patterns
- B) Increase radius
- C) Convert to RGB
- D) Detect edges

Answer: **A) Reduce number of patterns / use uniform patterns** ✅
Evidence (nli): In the LBP code method='uniform' uses uniform patterns, which reduces the number of patterns (n_points + 2 histogram bins). — `fact:0043 lab_04_manual:p9`

**41. In LBP code, radius = 1 defines:**

- A) Cell size
- B) Radius of circular neighborhood
- C) Number of bins
- D) Threshold

Answer: **B) Radius of circular neighborhood** ✅
Evidence (nli): In the LBP code radius = 1 is the radius of the circular neighborhood and n_points = 8 * radius is the number of neighboring pixels to consider. — `fact:0042 lab_04_manual:p8`

**42. In LBP code, n_points = 8 * radius defines:**

- A) Number of neighboring pixels to consider
- B) Image width
- C) Histogram bins
- D) Kernel size

Answer: **A) Number of neighboring pixels to consider** ✅
Evidence (nli): In the LBP code radius = 1 is the radius of the circular neighborhood and n_points = 8 * radius is the number of neighboring pixels to consider. — `fact:0042 lab_04_manual:p8`

**43. LBP is particularly useful for:**

- A) Texture classification
- B) Pedestrian detection only
- C) Audio recognition
- D) Depth estimation

Answer: **A) Texture classification** ✅
Evidence (nli): LBP is particularly useful for texture classification and face recognition. — `fact:0034 lab_04_manual:p7`

**44. LBP captures local patterns by:**

- A) Comparing intensity of a pixel with its neighbors
- B) Computing Fourier transform
- C) Applying Canny
- D) Color quantization

Answer: **A) Comparing intensity of a pixel with its neighbors** ✅
Evidence (nli): LBP captures local patterns by comparing the intensity of a pixel with its neighboring pixels. — `fact:0033 lab_04_manual:p7`

**45. LBP histogram counts occurrences of:**

- A) Colors
- B) Each LBP pattern
- C) Gradients
- D) Edges

Answer: **B) Each LBP pattern** ✅
Evidence (nli): The LBP histogram counts the occurrences of each LBP pattern in the image. — `fact:0039 lab_04_manual:p8`

**46. Histogram of Color captures:**

- A) Distribution of color intensities
- B) Edge directions
- C) Texture energy
- D) Gradient magnitude

Answer: **A) Distribution of color intensities** ✅
Evidence (nli): Histogram of Color captures the distribution of color intensities in an image. — `fact:0045 lab_04_manual:p9`

**47. Histogram of Color divides color space into:**

- A) Bins
- B) Blocks
- C) Cells
- D) Layers

Answer: **A) Bins** ✅
Evidence (nli): Histogram of Color divides the color space (RGB or HSV) into bins and counts the number of pixels in each bin. — `fact:0046 lab_04_manual:p9`

**48. Common color spaces for Histogram of Color:**

- A) RGB or HSV
- B) Only grayscale
- C) Only binary
- D) Only CMYK

Answer: **A) RGB or HSV** ✅
Evidence (nli): Histogram of Color divides the color space (RGB or HSV) into bins and counts the number of pixels in each bin. — `fact:0046 lab_04_manual:p9`

**49. Histogram of Color counts:**

- A) Number of pixels in each bin
- B) Number of edges
- C) Number of corners
- D) Number of gradients

Answer: **A) Number of pixels in each bin** ✅
Evidence (nli): Histogram of Color divides the color space (RGB or HSV) into bins and counts the number of pixels in each bin. — `fact:0046 lab_04_manual:p9`

**50. Histogram of Color is an example of:**

- A) Local feature
- B) Global feature
- C) Edge detector
- D) Filter

Answer: **B) Global feature** ✅
Evidence (nli): Histogram of Color (a color histogram) is an example of a global feature because it is computed over the entire image. — `fact:0047 lab_04_manual:p5`

**51. HED stands for:**

- A) Histogram of Edge Detection
- B) Histogram of Edge Directions
- C) High Edge Density
- D) Histogram of Edge Distribution

Answer: **B) Histogram of Edge Directions** ✅
Evidence (nli): HED stands for Histogram of Edge Directions. — `fact:0048 lab_04_manual:p9`

**52. HED captures:**

- A) Distribution of edge orientations
- B) Color distribution
- C) Texture energy
- D) Pixel coordinates

Answer: **A) Distribution of edge orientations** ✅
Evidence (nli): HED captures the distribution of edge orientations and the dominant edge directions in an image. — `fact:0049 lab_04_manual:p9`

**53. HED is useful for:**

- A) Texture analysis, object recognition, image segmentation
- B) Audio processing
- C) File compression
- D) Printing

Answer: **A) Texture analysis, object recognition, image segmentation** ✅
Evidence (nli): HED is useful for texture analysis, object recognition and image segmentation. — `fact:0050 lab_04_manual:p9`

**54. In HED, edge detection can use:**

- A) Only Canny
- B) Canny, Sobel, or Scharr
- C) Only K-means
- D) Only PCA

Answer: **B) Canny, Sobel, or Scharr** ✅
Evidence (nli): HED edge detection can use the Canny edge detector, the Sobel operator or the Scharr operator, producing an edge map. — `fact:0052 lab_04_manual:p10`

**55. In HED, orientation at each pixel is calculated using:**

- A) Arctangent function
- B) Sine function
- C) Cosine function
- D) Logarithm

Answer: **A) Arctangent function** ✅
Evidence (nli): In HED the orientation (angle) of the gradient at each pixel is calculated using the arctangent function. — `fact:0053 lab_04_manual:p10`

**56. In HED, 360 degrees can be divided into:**

- A) 4 bins
- B) 8 bins for octagonal directions
- C) 16 bins
- D) 32 bins

Answer: **B) 8 bins for octagonal directions** ✅
Evidence (nli): In HED orientation binning, 360 degrees is divided into 8 bins for octagonal directions. — `fact:0054 lab_04_manual:p10`

**57. HED feature vector is:**

- A) Histogram of edge directions
- B) Color histogram
- C) HOG vector
- D) LBP histogram

Answer: **A) Histogram of edge directions** ✅
Evidence (nli): The HED feature vector is the histogram of edge directions, counting pixels in each orientation bin. — `fact:0055 lab_04_manual:p10`

**58. In HED code, gradient orientation is computed by:**

- A) np.arctan2(gradient_x, gradient_y) * 180 / np.pi
- B) np.arctan2(gradient_y, gradient_x) * 180 / np.pi
- C) np.tan(gradient_y/gradient_x)
- D) np.sin(gradient_x)

Answer: **B) np.arctan2(gradient_y, gradient_x) * 180 / np.pi** ✅
Evidence (nli): In the HED code gradient orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi, with y first then x. — `fact:0056 lab_04_manual:p11`

**59. HED histogram bins in code:**

- A) 4
- B) 8
- C) 16
- D) 256

Answer: **B) 8** ✅
Evidence (triple): HED histogram | histogram bins in code | 8 — `triple:0128 lab_04_manual:p11`

**60. HED histogram range in code:**

- A) (0, 180)
- B) (0, 360)
- C) (0, 90)
- D) (0, 255)

Answer: **B) (0, 360)** ✅
Evidence (nli): The HED code builds the histogram with np.histogram(gradient_orientation, bins=8, range=(0, 360)): 8 bins over the range 0 to 360. — `fact:0057 lab_04_manual:p11`

**61. HIG stands for:**

- A) Histogram of Image Gradients
- B) Histogram of Intensity Gradients
- C) High Intensity Gradient
- D) Histogram of Intensity Groups

Answer: **B) Histogram of Intensity Gradients** ✅
Evidence (nli): HIG stands for Histogram of Intensity Gradients. — `fact:0059 lab_04_manual:p11`

**62. HIG captures:**

- A) Distribution of intensity gradients
- B) Color distribution
- C) Edge directions only
- D) Texture energy

Answer: **A) Distribution of intensity gradients** ✅
Evidence (nli): HIG stands for Histogram of Intensity Gradients. — `fact:0059 lab_04_manual:p11`

**63. HIG is useful for:**

- A) Object recognition, texture analysis, image classification
- B) Audio processing
- C) Printing
- D) File compression

Answer: **A) Object recognition, texture analysis, image classification** ✅
Evidence (nli): HIG is useful for object recognition, texture analysis and image classification. — `fact:0061 lab_04_manual:p11`

**64. Gradient magnitude formula:**

- A) G = dx + dy
- B) G = sqrt(dx^2 + dy^2)
- C) G = dx^2 - dy^2
- D) G = dx * dy

Answer: **B) G = sqrt(dx^2 + dy^2)** ✅
Evidence (nli): Gradient magnitude formula: G = sqrt(dx^2 + dy^2). — `fact:0063 lab_04_manual:p12`

**65. Gradient orientation formula:**

- A) Θ = atan2(dx, dy)
- B) Θ = atan2(dy, dx)
- C) Θ = sin(dx/dy)
- D) Θ = cos(dy/dx)

Answer: **B) Θ = atan2(dy, dx)** ✅
Evidence (nli): Gradient orientation formula: theta = atan2(dy, dx), with dy first then dx. — `fact:0064 lab_04_manual:p12`

**66. In HIG, gradient can be computed using:**

- A) Sobel, Scharr, or other gradient methods
- B) Only Canny
- C) Only K-means
- D) Only PCA

Answer: **A) Sobel, Scharr, or other gradient methods** ✅
Evidence (nli): HIG computes the gradient with the Sobel operator, Scharr operator or other gradient methods, separately for horizontal dx and vertical dy. — `fact:0062 lab_04_manual:p12`

**67. In HIG, histogram range for orientations is typically:**

- A) 0 to 180 degrees
- B) 0 to 360 degrees
- C) 0 to 90 degrees
- D) 0 to 255 degrees

Answer: **B) 0 to 360 degrees** ✅
Evidence (nli): In HIG the range of gradient orientations is typically 0 to 360 degrees, divided into bins. — `fact:0065 lab_04_manual:p12`

**68. HIG histogram can be normalized to make it:**

- A) Scale-invariant
- B) Colorful
- C) Binary
- D) Noisy

Answer: **A) Scale-invariant** ✅
Evidence (nli): The HIG histogram can optionally be normalized to make it scale-invariant. — `fact:0066 lab_04_manual:p12`

**69. HIG feature vector is:**

- A) Histogram of gradient orientations
- B) Color histogram
- C) LBP histogram
- D) HOG block

Answer: **A) Histogram of gradient orientations** ✅
Evidence (nli): The HIG feature vector is the histogram of gradient orientations, capturing intensity gradient information. — `fact:0067 lab_04_manual:p12`

**70. HIG is similar to HOG but usually:**

- A) Global instead of local cells/blocks
- B) Only for audio
- C) Only for video
- D) Only for depth

Answer: **A) Global instead of local cells/blocks** ✅
Evidence (nli): HIG is similar to HOG but is usually computed globally over the whole image instead of local cells and blocks. — `fact:0068 lab_04_manual:p12`

**71. Texture energy is computed by:**

- A) Sum of pixel values
- B) Sum of squared pixel values in neighborhood
- C) Average of pixel values
- D) Product of pixel values

Answer: **B) Sum of squared pixel values in neighborhood** ✅
Evidence (nli): The texture code computes energy with cv2.filter2D(image**2, -1, np.ones((3,3))), which sums squared pixel values in the neighborhood. — `fact:0080 lab_04_manual:p14`

**72. Texture contrast is computed as:**

- A) Mean of pixel values
- B) Standard deviation of pixel values in neighborhood
- C) Sum of squared pixels
- D) Maximum pixel value

Answer: **B) Standard deviation of pixel values in neighborhood** ✅
Evidence (nli): Texture contrast is the standard deviation of pixel intensities in the neighborhood: C = sqrt(Var(pixel values)). — `fact:0073 lab_04_manual:p13`

**73. Texture contrast formula in manual:**

- A) C = var(pixel values)
- B) C = sqrt(var(pixel values))
- C) C = sum(pixel values)
- D) C = max(pixel values)

Answer: **B) C = sqrt(var(pixel values))** ✅
Evidence (nli): Texture contrast is the standard deviation of pixel intensities in the neighborhood: C = sqrt(Var(pixel values)). — `fact:0073 lab_04_manual:p13`

**74. Common neighborhood sizes for texture analysis:**

- A) 3x3 or 5x5
- B) 8x8 or 16x16
- C) 32x32 or 64x64
- D) 1x1 or 2x2

Answer: **A) 3x3 or 5x5** ✅
Evidence (nli): Common neighborhood (window) sizes for texture analysis are 3x3 or 5x5. — `fact:0075 lab_04_manual:p14`

**75. Texture energy histogram captures:**

- A) Distribution of texture energy values
- B) Distribution of colors
- C) Distribution of edges
- D) Distribution of gradients

Answer: **A) Distribution of texture energy values** ✅
Evidence (nli): The texture energy histogram captures the distribution of texture energy values. — `fact:0076 lab_04_manual:p14`

**76. Texture contrast histogram captures:**

- A) Distribution of contrast values
- B) Distribution of colors
- C) Distribution of orientations
- D) Distribution of gradients

Answer: **A) Distribution of contrast values** ✅
Evidence (nli): The texture contrast histogram captures the distribution of contrast values. — `fact:0077 lab_04_manual:p14`

**77. Texture energy and contrast features can be used for:**

- A) Texture classification, segmentation
- B) Audio processing
- C) File compression
- D) Printing

Answer: **A) Texture classification, segmentation** ✅
Evidence (nli): Texture energy and contrast features are used for texture classification and segmentation. — `fact:0079 lab_04_manual:p14`

**78. In texture code, energy uses cv2.filter2D(image**2, -1, np.ones((3,3))) because:**

- A) It sums squared pixel values in neighborhood
- B) It blurs image
- C) It detects edges
- D) It converts to grayscale

Answer: **A) It sums squared pixel values in neighborhood** ✅
Evidence (nli): The texture code computes energy with cv2.filter2D(image**2, -1, np.ones((3,3))), which sums squared pixel values in the neighborhood. — `fact:0080 lab_04_manual:p14`

**79. In texture code, histograms are created using:**

- A) np.histogram
- B) cv2.histogram
- C) plt.hist
- D) skimage.histogram

Answer: **A) np.histogram** ✅
Evidence (nli): The texture code creates histograms with np.histogram (bins=256) and normalizing the histograms is optional. — `fact:0081 lab_04_manual:p15`

**80. Normalizing texture histograms is:**

- A) Required
- B) Optional
- C) Impossible
- D) Only for color

Answer: **B) Optional** ✅
Evidence (nli): The texture code creates histograms with np.histogram (bins=256) and normalizing the histograms is optional. — `fact:0081 lab_04_manual:p15`

**81. Filtering involves applying a ___ to an image.**

- A) Small matrix called kernel/filter
- B) Large database
- C) Color space
- D) Histogram

Answer: **A) Small matrix called kernel/filter** ✅
Evidence (nli): Filtering applies a filter or kernel, a small matrix of numbers, to an input image. — `fact:0084 lab_04_manual:p16`

**82. Each element of the filter represents:**

- A) Weighted contribution to new pixel value
- B) Pixel coordinate
- C) Color channel
- D) Edge direction

Answer: **A) Weighted contribution to new pixel value** ✅
Evidence (nli): Each element of the filter represents a weighted contribution to the new pixel value. — `fact:0085 lab_04_manual:p16`

**83. Convolution involves:**

- A) Sliding filter over image, multiplying, summing
- B) Sliding filter over image, element-wise multiplying, summing
- C) Sorting pixels
- D) Averaging colors

Answer: **B) Sliding filter over image, element-wise multiplying, summing** ✅
Evidence (nli): Convolution slides the filter over the image, element-wise multiplies the filter and the image region, and sums the results (a dot product). — `fact:0086 lab_04_manual:p16`

**84. Convolution result is called:**

- A) Output or convolved image
- B) Histogram
- C) Feature vector
- D) Kernel

Answer: **A) Output or convolved image** ✅
Evidence (nli): The result of convolution is a new image called the output or convolved image. — `fact:0087 lab_04_manual:p16`

**85. Box blur replaces each pixel with:**

- A) Average of neighboring pixels
- B) Maximum of neighbors
- C) Minimum of neighbors
- D) Median of neighbors

Answer: **A) Average of neighboring pixels** ✅
Evidence (nli): Box blur replaces each pixel with the average of its neighboring pixels within a square kernel; useful for noise reduction and smoothing. — `fact:0088 lab_04_manual:p16`

**86. Box blur kernel in manual:**

- A) np.ones((3,3))
- B) np.ones((3,3), dtype=np.float32) / 9
- C) np.eye(3)
- D) np.zeros((3,3))

Answer: **B) np.ones((3,3), dtype=np.float32) / 9** ✅
Evidence (nli): The box blur kernel in the manual is np.ones((3,3), dtype=np.float32) / 9 applied with cv2.filter2D(image, -1, kernel). — `fact:0089 lab_04_manual:p16`

**87. Gaussian blur uses:**

- A) Gaussian-shaped kernel
- B) Box kernel
- C) Sobel kernel
- D) Emboss kernel

Answer: **A) Gaussian-shaped kernel** ✅
Evidence (nli): Gaussian blur uses a Gaussian-shaped kernel and gives smoother results than box blur; often used for noise reduction. — `fact:0090 lab_04_manual:p17`

**88. Gaussian blur provides smoother results compared to:**

- A) Box blur
- B) Sobel
- C) Scharr
- D) Canny

Answer: **A) Box blur** ✅
Evidence (nli): Gaussian blur uses a Gaussian-shaped kernel and gives smoother results than box blur; often used for noise reduction. — `fact:0090 lab_04_manual:p17`

**89. Sobel and Scharr operators detect:**

- A) Edges by emphasizing rapid changes in pixel values
- B) Colors
- C) Textures only
- D) Corners only

Answer: **A) Edges by emphasizing rapid changes in pixel values** ✅
Evidence (nli): Sobel and Scharr operators detect edges by emphasizing rapid changes in pixel values. — `fact:0092 lab_04_manual:p17`

**90. Embossing creates:**

- A) 3D effect by emphasizing differences in neighboring pixel values
- B) Blur
- C) Edge map
- D) Histogram

Answer: **A) 3D effect by emphasizing differences in neighboring pixel values** ✅
Evidence (nli): Embossing creates a 3D effect by emphasizing differences in neighboring pixel values; used for artistic or stylized effects. — `fact:0093 lab_04_manual:p17`

**91. Standard convolution is also known as:**

- A) Full convolution
- B) Valid convolution
- C) Same convolution
- D) Strided convolution

Answer: **A) Full convolution** ✅
Evidence (nli): Standard convolution is also known as full convolution and is the most common type; it centers the kernel over each pixel. — `fact:0094 lab_04_manual:p17`

**92. Valid convolution (no padding) produces output image:**

- A) Same size as input
- B) Smaller than input
- C) Larger than input
- D) Infinite

Answer: **B) Smaller than input** ✅
Evidence (nli): Valid convolution (no padding) only processes pixels where the kernel fully overlaps the image and produces an output smaller than the input. — `fact:0095 lab_04_manual:p18`

**93. Same convolution adds:**

- A) Zero-padding
- B) Random noise
- C) Blur
- D) Color

Answer: **A) Zero-padding** ✅
Evidence (nli): Same convolution adds zero-padding so the output image has the same dimensions as the input, preventing information loss at the boundaries. — `fact:0096 lab_04_manual:p18`

**94. Same convolution ensures output image has:**

- A) Same dimensions as input
- B) Smaller dimensions
- C) Larger dimensions
- D) No dimensions

Answer: **A) Same dimensions as input** ✅
Evidence (nli): Same convolution adds zero-padding so the output image has the same dimensions as the input, preventing information loss at the boundaries. — `fact:0096 lab_04_manual:p18`

**95. Valid convolution with strides:**

- A) Skips some pixels based on specified stride
- B) Adds padding
- C) Keeps same size
- D) Blurs image

Answer: **A) Skips some pixels based on specified stride** ✅
Evidence (nli): Valid convolution with strides skips some pixels based on the specified stride, giving an output smaller than the input determined by the stride. — `fact:0097 lab_04_manual:p19`

**96. Edge detection identifies:**

- A) Boundaries within an image
- B) Colors
- C) Textures only
- D) Histograms

Answer: **A) Boundaries within an image** ✅
Evidence (nli): Edge detection identifies boundaries within an image, transitions from one object or region to another. — `fact:0098 lab_04_manual:p19`

**97. Edges correspond to changes in:**

- A) Color, intensity, or texture
- B) Only color
- C) Only intensity
- D) Only texture

Answer: **A) Color, intensity, or texture** ✅
Evidence (nli): Edges correspond to changes in color, intensity or texture. — `fact:0099 lab_04_manual:p19`

**98. Gradient-based edge detection uses:**

- A) Gradient of pixel intensities
- B) Histogram
- C) Color space
- D) Fourier transform

Answer: **A) Gradient of pixel intensities** ✅
Evidence (nli): Gradient-based edge detection calculates the gradient (rate of change) of pixel intensities. — `fact:0101 lab_04_manual:p20`

**99. Common gradient-based operators:**

- A) Sobel, Canny, LBP
- B) Sobel, Prewitt, Scharr
- C) HOG, LBP, HED
- D) K-means, PCA, SVM

Answer: **B) Sobel, Prewitt, Scharr** ✅
Evidence (nli): Common gradient-based operators are Sobel, Prewitt and Scharr. — `fact:0103 lab_04_manual:p20`

**100. Canny edge detector was developed by:**

- A) John F. Canny
- B) David Marr
- C) Sobel
- D) Prewitt

Answer: **A) John F. Canny** ✅
Evidence (triple): Canny | developed by | John F. Canny — `triple:0095 lab_04_manual:p21`

**101. Canny edge detector was developed in:**

- A) 1976
- B) 1986
- C) 1996
- D) 2006

Answer: **B) 1986** ✅
Evidence (triple): Canny | developed in | 1986 — `triple:0094 lab_04_manual:p21`

**102. First step in Canny:**

- A) Gaussian smoothing
- B) Non-maximum suppression
- C) Hysteresis
- D) Gradient calculation

Answer: **A) Gaussian smoothing** ✅
Evidence (nli): Canny step 1 (first step): Gaussian smoothing reduces noise by convolving a Gaussian kernel with the image. — `fact:0112 lab_04_manual:p22`

**103. Gaussian smoothing in Canny reduces:**

- A) Noise
- B) Edges
- C) Color
- D) Resolution

Answer: **A) Noise** ✅
Evidence (nli): Gaussian smoothing in Canny reduces noise to prevent false edges; non-maximum suppression thins thick edges to one-pixel-wide lines. — `fact:0319 lab_manual_06:p12`

**104. Second step in Canny:**

- A) Gaussian smoothing
- B) Gradient calculation
- C) Non-maximum suppression
- D) Hysteresis

Answer: **B) Gradient calculation** ✅
Evidence (nli): Canny step 2 (second step): gradient calculation with two 3x3 Sobel kernels gives gradient magnitude and direction. — `fact:0113 lab_04_manual:p22`

**105. Canny gradient calculation uses:**

- A) Sobel operators
- B) LBP
- C) HOG
- D) K-means

Answer: **A) Sobel operators** ✅
Evidence (nli): Canny step 2 (second step): gradient calculation with two 3x3 Sobel kernels gives gradient magnitude and direction. — `fact:0113 lab_04_manual:p22`

**106. Non-maximum suppression in Canny:**

- A) Thins edges by keeping local maxima
- B) Blurs edges
- C) Adds noise
- D) Colors edges

Answer: **A) Thins edges by keeping local maxima** ✅
Evidence (nli): Canny step 3 (third step): non-maximum suppression keeps local maxima along the gradient direction and thins the edges. — `fact:0114 lab_04_manual:p22`

**107. Hysteresis in Canny uses:**

- A) One threshold
- B) Two thresholds
- C) Three thresholds
- D) No thresholds

Answer: **B) Two thresholds** ✅
Evidence (nli): Canny step 4 (final step): edge tracking by hysteresis uses two thresholds, a high threshold and a low threshold. — `fact:0115 lab_04_manual:p22`

**108. In Canny, pixels above high threshold are:**

- A) Strong edge points
- B) Weak edge points
- C) Non-edge points
- D) Noise

Answer: **A) Strong edge points** ✅
Evidence (triple): high threshold | in Canny used to | detect strong edges — `triple:0042 lab_05_manual:p10`

**109. In Canny, pixels between low and high threshold are:**

- A) Strong edge points
- B) Weak edge points
- C) Non-edge points
- D) Noise

Answer: **B) Weak edge points** ✅
Evidence (triple): pixels between low and high threshold | are in Canny | weak edge points — `triple:0127 lab_04_manual:p22`

**110. In Canny, weak edge points are kept if:**

- A) Connected to strong edge points
- B) Isolated
- C) Above high threshold
- D) Below low threshold

Answer: **A) Connected to strong edge points** ✅
Evidence (nli): In Canny, weak edge points are kept only if connected to strong edge points; pixels below the low threshold are rejected as non-edge points. — `fact:0118 lab_04_manual:p22`

**111. Canny parameter sigma controls:**

- A) Gaussian filter kernel size / smoothing
- B) Threshold
- C) Gradient direction
- D) Color space

Answer: **A) Gaussian filter kernel size / smoothing** ✅
Evidence (nli): The Canny parameter sigma sets the Gaussian kernel size and smoothing; a larger sigma gives a smoother image but loses fine details. — `fact:0119 lab_04_manual:p23`

**112. Larger sigma in Canny results in:**

- A) Smoother image but loss of fine details
- B) Sharper edges
- C) More noise
- D) No edges

Answer: **A) Smoother image but loss of fine details** ✅
Evidence (nli): The Canny parameter sigma sets the Gaussian kernel size and smoothing; a larger sigma gives a smoother image but loses fine details. — `fact:0119 lab_04_manual:p23`

**113. Canny threshold1 in OpenCV is:**

- A) Low threshold
- B) High threshold
- C) Sigma
- D) Aperture size

Answer: **A) Low threshold** ✅
Evidence (triple): threshold1 | in cv2.Canny is | low threshold — `triple:0040 lab_04_manual:p23`

**114. Canny threshold2 in OpenCV is:**

- A) Low threshold
- B) High threshold
- C) Sigma
- D) Aperture size

Answer: **B) High threshold** ✅
Evidence (triple): threshold2 | in cv2.Canny is | high threshold — `triple:0041 lab_04_manual:p23`

**115. Canny code uses function:**

- A) cv2.canny(image, 100, 200)
- B) cv2.Canny(image, threshold1=100, threshold2=200)
- C) cv2.edge(image, 100, 200)
- D) cv2.CannyEdge(image, 100, 200)

Answer: **B) cv2.Canny(image, threshold1=100, threshold2=200)** ✅
Evidence (nli): The Canny code is cv2.Canny(image, threshold1=100, threshold2=200). — `fact:0122 lab_04_manual:p23`

**116. LoG stands for:**

- A) Laplacian of Gaussian
- B) Logarithm of Gradient
- C) Local of Gradient
- D) Laplacian of Gradient

Answer: **A) Laplacian of Gaussian** ✅
Evidence (nli): LoG stands for Laplacian of Gaussian; it combines Gaussian smoothing and Laplacian edge detection and highlights zero-crossings of the second derivative. — `fact:0105 lab_04_manual:p20`

**117. LoG combines:**

- A) Gaussian smoothing and Laplacian edge detection
- B) Sobel and Prewitt
- C) HOG and LBP
- D) Canny and Hysteresis

Answer: **A) Gaussian smoothing and Laplacian edge detection** ✅
Evidence (nli): LoG stands for Laplacian of Gaussian; it combines Gaussian smoothing and Laplacian edge detection and highlights zero-crossings of the second derivative. — `fact:0105 lab_04_manual:p20`

**118. Prewitt operator uses:**

- A) 3x3 kernels for horizontal and vertical gradients
- B) 5x5 kernels
- C) 7x7 kernels
- D) 1x1 kernels

Answer: **A) 3x3 kernels for horizontal and vertical gradients** ✅
Evidence (nli): The Prewitt operator is a gradient-based method using 3x3 kernels for horizontal and vertical gradients. — `fact:0107 lab_04_manual:p20`

**119. Marr-Hildreth edge detector uses:**

- A) LoG after Gaussian smoothing
- B) Sobel only
- C) Prewitt only
- D) Canny only

Answer: **A) LoG after Gaussian smoothing** ✅
Evidence (nli): The Marr-Hildreth edge detector uses the LoG operator after Gaussian smoothing and locates edges at zero-crossings. — `fact:0108 lab_04_manual:p21`

**120. Sobel-Feldman operator emphasizes:**

- A) Diagonal edges in addition to horizontal and vertical edges
- B) Only horizontal edges
- C) Only vertical edges
- D) Only color edges

Answer: **A) Diagonal edges in addition to horizontal and vertical edges** ✅
Evidence (nli): The manual describes the Sobel-Feldman operator as a variation of Sobel emphasizing diagonal edges in addition to horizontal and vertical edges. — `fact:0109 lab_04_manual:p21`

**121. CNN-based edge detection can:**

- A) Learn complex edge patterns and adapt to domains
- B) Only detect straight lines
- C) Only detect circles
- D) Only detect colors

Answer: **A) Learn complex edge patterns and adapt to domains** ✅
Evidence (nli): CNN-based edge detection learns complex edge patterns and adapts to various image domains. — `fact:0110 lab_04_manual:p21`

**122. Edge detection reduces data while:**

- A) Highlighting essential features
- B) Removing all features
- C) Adding noise
- D) Changing colors

Answer: **A) Highlighting essential features** ✅
Evidence (nli): Edge detection is a preprocessing step that reduces the amount of data while highlighting essential features. — `fact:0100 lab_04_manual:p19`

**123. Gradient magnitude high indicates:**

- A) Edge likely present
- B) No edge
- C) Smooth region
- D) Color change only

Answer: **A) Edge likely present** ✅
Evidence (nli): A high gradient magnitude indicates an edge is likely present; edges typically occur where gradient magnitude is high. — `fact:0102 lab_04_manual:p20`

**124. Scharr operator is:**

- A) A gradient-based edge detector
- B) Not gradient-based
- C) Only for color
- D) Only for texture

Answer: **A) A gradient-based edge detector** ✅
Evidence (nli): Sobel and Scharr are gradient-based edge detectors approximating the image gradient in horizontal and vertical directions. — `fact:0106 lab_04_manual:p20`

**125. Canny is known for:**

- A) Accuracy and noise reduction
- B) Speed only
- C) Color detection
- D) Texture only

Answer: **A) Accuracy and noise reduction** ✅
Evidence (nli): Canny is a multi-stage edge detector known for its accuracy and noise reduction capabilities. — `fact:0104 lab_04_manual:p20`
