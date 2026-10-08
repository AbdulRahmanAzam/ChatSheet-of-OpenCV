# Answer sheet: mcq_lab05.txt

Graded 98/103 correct. Wrong: Q7, Q9, Q11, Q43, Q67

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. What is the main topic of Lab 05?**

- A) Image Classification
- B) Image Segmentation
- C) Object Detection
- D) Image Compression

Answer: **B) Image Segmentation** ✅
Evidence (triple): Lab 05 | main topic | Image Segmentation — `triple:0099 lab_05_manual:p1`

**2. What is the course code for this lab?**

- A) AI-4002
- B) AI-4004
- C) CS-4002
- D) CV-4002

Answer: **A) AI-4002** ✅
Evidence (triple): course code | course code | AI-4002 — `triple:0101 lab_01_manual:p1`

**3. Which university is this lab manual from?**

- A) NUST
- B) FAST
- C) NUCES Karachi
- D) IBA

Answer: **C) NUCES Karachi** ✅
Evidence (triple): university | from university | NUCES Karachi — `triple:0100 lab_01_manual:p1`

**4. Image segmentation partitions an image into:**

- A) Overlapping regions
- B) Distinct, non-overlapping regions
- C) Random pixels
- D) Compressed blocks

Answer: **B) Distinct, non-overlapping regions** ✅
Evidence (nli): Image segmentation partitions an image into distinct, non-overlapping regions, each corresponding to a meaningful object or part of the scene. — `fact:0245 lab_05_manual:p3`

**5. Each segment in image segmentation corresponds to:**

- A) A random color
- B) A meaningful object or part of the scene
- C) A single pixel
- D) The entire image

Answer: **B) A meaningful object or part of the scene** ✅
Evidence (nli): Image segmentation partitions an image into distinct, non-overlapping regions, each corresponding to a meaningful object or part of the scene. — `fact:0245 lab_05_manual:p3`

**6. The goal of image segmentation is to separate an image into areas that share similar:**

- A) File sizes
- B) Compression ratios
- C) Visual characteristics such as color, texture, or intensity
- D) Metadata

Answer: **C) Visual characteristics such as color, texture, or intensity** ✅
Evidence (nli): Segmentation separates an image into areas sharing similar visual characteristics such as color, texture or intensity. — `fact:0246 lab_05_manual:p3`

**7. Which of the following is NOT a visual characteristic used in segmentation?**

- A) Color
- B) Texture
- C) Intensity
- D) File size

Answer: **B) Texture** ❌ (key D: File size)
Evidence (nli): Visual characteristics used to group pixels in segmentation are color, texture and intensity; file size and metadata are not visual characteristics. — `fact:0518 lab_05_manual:p3`

**8. Preprocessing and feature extraction in segmentation includes:**

- A) Pixel values, edge detection, texture analysis, color similarity
- B) File compression, encryption
- C) Audio processing
- D) Text recognition

Answer: **A) Pixel values, edge detection, texture analysis, color similarity** ✅
Evidence (triple): preprocessing and feature extraction | includes in segmentation | pixel values, edge detection, texture analysis, color similarity — `triple:0105 lab_05_manual:p3`

**9. Which of the following is a segmentation algorithm?**

- A) Region growing
- B) Clustering
- C) Watershed
- D) All of the above

Answer: **A) Region growing** ❌ (key D: All of the above)
Evidence (nli): In the Lab 05 landscape figure, segmentation algorithms shown are region growing, clustering (K-means), watershed and a deep learning model (CNN). — `fact:0513 lab_05_manual:p3`

**10. In the landscape example, the segmented region map includes:**

- A) Buildings, roads, cars
- B) Sky, mountains, water
- C) People, animals, plants
- D) Text, numbers, symbols

Answer: **B) Sky, mountains, water** ✅
Evidence (triple): landscape example | segmented region map includes | sky, mountains, water — `triple:0102 lab_05_manual:p3`

**11. In the portrait example, which region is NOT shown?**

- A) Hair region
- B) Face region
- C) Clothing region
- D) Background region

Answer: **B) Face region** ❌ (key D: Background region)
Evidence (nli): Segmentation divides an image into meaningful segments or regions, often separating objects from the background. — `fact:0137 lab_01_manual:p5`

**12. In the street scene example, the segments include:**

- A) Car, tree, person
- B) Sky, water, mountain
- C) Hair, face, clothing
- D) Cell nuclei, cells

Answer: **A) Car, tree, person** ✅
Evidence (triple): street scene example | segments include | car, tree, person — `triple:0103 lab_05_manual:p4`

**13. Segmentation is the process of dividing an image into:**

- A) Random parts
- B) Meaningful, separate parts
- C) Compressed blocks
- D) Binary digits

Answer: **B) Meaningful, separate parts** ✅
Evidence (nli): Segmentation divides an image into meaningful segments or regions, often separating objects from the background. — `fact:0137 lab_01_manual:p5`

**14. The mathematics behind image segmentation includes:**

- A) Thresholding, edge detection, clustering, graph theory
- B) Calculus, algebra, geometry
- C) Sorting, searching, hashing
- D) Encryption, decryption

Answer: **A) Thresholding, edge detection, clustering, graph theory** ✅
Evidence (nli): In graph-based segmentation pixels are nodes and edges represent pixel similarity; minimum spanning trees or graph cuts partition the image. — `fact:0251 lab_05_manual:p5`

**15. Thresholding compares a pixel's intensity value to:**

- A) A random value
- B) A predefined threshold
- C) The image mean
- D) The maximum pixel value

Answer: **B) A predefined threshold** ✅
Evidence (nli): In thresholding segmentation a pixel intensity is compared to a predefined threshold: above it belongs to one region (object), otherwise to another (background). — `fact:0248 lab_05_manual:p5`

**16. Edge detection algorithms identify:**

- A) Slow changes in intensity
- B) Rapid changes in intensity
- C) Constant intensity
- D) Random noise

Answer: **B) Rapid changes in intensity** ✅
Evidence (nli): Edge-based segmentation identifies rapid intensity changes at object boundaries using Sobel, Canny or Laplacian of Gaussian (LoG). — `fact:0249 lab_05_manual:p5`

**17. Which is a common edge detection method?**

- A) Sobel
- B) Canny
- C) Laplacian of Gaussian (LoG)
- D) All of the above

Answer: **D) All of the above** ✅
Evidence (nli): HED edge detection can use the Canny edge detector, the Sobel operator or the Scharr operator, producing an edge map. — `fact:0052 lab_04_manual:p10`

**18. Clustering techniques group similar pixels together based on:**

- A) File size
- B) Feature similarity
- C) Metadata
- D) Compression ratio

Answer: **B) Feature similarity** ✅
Evidence (nli): Clustering-based segmentation groups pixels into clusters by feature similarity using K-Means, Mean-Shift or DBSCAN. — `fact:0283 lab_05_manual:p14`

**19. Which of the following is a clustering technique?**

- A) K-Means
- B) Mean-Shift
- C) DBSCAN
- D) All of the above

Answer: **D) All of the above** ✅
Evidence (nli): K-means repeats assignment and update until convergence, when cluster assignments no longer change significantly; each cluster becomes a segment. — `fact:0287 lab_05_manual:p14`

**20. In graph-based segmentation, pixels are represented as:**

- A) Edges
- B) Nodes
- C) Weights
- D) Clusters

Answer: **B) Nodes** ✅
Evidence (nli): In graph-based segmentation pixels are nodes and edges represent pixel similarity; minimum spanning trees or graph cuts partition the image. — `fact:0251 lab_05_manual:p5`

**21. In graph-based segmentation, edges between nodes represent:**

- A) Distance
- B) Similarity between pixel values
- C) File size
- D) Color depth

Answer: **B) Similarity between pixel values** ✅
Evidence (nli): In graph-based segmentation pixels are nodes and edges represent pixel similarity; minimum spanning trees or graph cuts partition the image. — `fact:0251 lab_05_manual:p5`

**22. Which graph algorithms can be used for segmentation?**

- A) Minimum spanning trees
- B) Graph cuts
- C) Both A and B
- D) Neither A nor B

Answer: **C) Both A and B** ✅
Evidence (nli): In matrix [[a, b], [c, d]], c (Y of the new X-axis) gives vertical shearing and b (X of the new Y-axis) gives horizontal shearing. — `fact:0220 lab_03_manual:p10`

**23. Medical imaging segmentation is used to:**

- A) Identify anatomical structures or tumors
- B) Detect pedestrians
- C) Classify land cover
- D) Track objects in AR

Answer: **A) Identify anatomical structures or tumors** ✅
Evidence (nli): Segmentation in medical imaging isolates anatomical structures or tumors in X-rays, CT scans or MRIs. — `fact:0253 lab_05_manual:p6`

**24. Autonomous vehicles use segmentation to detect:**

- A) Tumors
- B) Cell nuclei
- C) Pedestrians, vehicles, and road signs
- D) Deforestation

Answer: **C) Pedestrians, vehicles, and road signs** ✅
Evidence (nli): In autonomous vehicles segmentation detects pedestrians, other vehicles and road signs. — `fact:0254 lab_05_manual:p6`

**25. Satellite imagery segmentation helps to:**

- A) Count cells
- B) Classify land cover, monitor deforestation, assess urban growth
- C) Detect tumors
- D) Track objects in real-time video

Answer: **B) Classify land cover, monitor deforestation, assess urban growth** ✅
Evidence (nli): Satellite image segmentation classifies land cover, monitors deforestation and assesses urban growth. — `fact:0255 lab_05_manual:p6`

**26. Biomedical imaging segmentation is used to:**

- A) Count cells, identify cell nuclei, diagnose diseases
- B) Detect road signs
- C) Classify land cover
- D) Track objects in AR

Answer: **A) Count cells, identify cell nuclei, diagnose diseases** ✅
Evidence (nli): In biomedical microscopy segmentation counts cells, identifies cell nuclei and helps diagnose diseases. — `fact:0256 lab_05_manual:p6`

**27. Augmented Reality (AR) uses segmentation for:**

- A) Scene understanding and real-time object tracking
- B) Tumor detection
- C) Deforestation monitoring
- D) Cell counting

Answer: **A) Scene understanding and real-time object tracking** ✅
Evidence (nli): In augmented reality segmentation supports scene understanding and real-time object tracking. — `fact:0257 lab_05_manual:p6`

**28. Thresholding divides an image into regions based on:**

- A) Color depth
- B) Pixel intensity values
- C) File format
- D) Image resolution

Answer: **B) Pixel intensity values** ✅
Evidence (nli): Thresholding divides an image into two regions by pixel intensity; it is simple and effective for objects with distinct intensity differences. — `fact:0258 lab_05_manual:p8`

**29. Global thresholding applies:**

- A) Different thresholds to different regions
- B) A single global threshold to the entire image
- C) No threshold
- D) Adaptive thresholds

Answer: **B) A single global threshold to the entire image** ✅
Evidence (nli): Global thresholding applies a single threshold value to the entire image; effective when objects and background have a clear intensity difference. — `fact:0259 lab_05_manual:p8`

**30. Adaptive thresholding uses:**

- A) A single threshold
- B) Different threshold values for different regions
- C) Random thresholds
- D) No threshold

Answer: **B) Different threshold values for different regions** ✅
Evidence (nli): Adaptive thresholding uses different threshold values for different regions, adapting to local intensity variations; ideal for varying lighting conditions. — `fact:0261 lab_05_manual:p8`

**31. Adaptive thresholding is ideal for images with:**

- A) Uniform lighting
- B) Varying lighting conditions
- C) No lighting
- D) High contrast

Answer: **B) Varying lighting conditions** ✅
Evidence (nli): Adaptive thresholding uses different threshold values for different regions, adapting to local intensity variations; ideal for varying lighting conditions. — `fact:0261 lab_05_manual:p8`

**32. Otsu's thresholding automatically selects an optimal threshold to maximize:**

- A) Intra-class variance
- B) Inter-class variance
- C) Image contrast
- D) Noise

Answer: **B) Inter-class variance** ✅
Evidence (nli): Otsu's thresholding automatically selects an optimal threshold that maximizes the inter-class variance between object and background pixels. — `fact:0263 lab_05_manual:p9`

**33. Otsu's thresholding is suitable when the distribution of pixel intensities is:**

- A) Uniform
- B) Bimodal
- C) Random
- D) Skewed

Answer: **B) Bimodal** ✅
Evidence (triple): Otsu | suitable distribution | bimodal — `triple:0070 lab_05_manual:p9`

**34. Color-based thresholding applies thresholding in:**

- A) Only grayscale
- B) Multiple color channels (e.g., RGB, HSV)
- C) Only binary
- D) Only alpha channel

Answer: **B) Multiple color channels (e.g., RGB, HSV)** ✅
Evidence (nli): Color-based thresholding thresholds multiple color channels (RGB or HSV) to segment objects with distinct colors. — `fact:0266 lab_05_manual:p9`

**35. Color-based thresholding is useful for segmenting objects with:**

- A) Similar intensity
- B) Distinct colors
- C) No color
- D) High noise

Answer: **B) Distinct colors** ✅
Evidence (nli): Color-based thresholding thresholds multiple color channels (RGB or HSV) to segment objects with distinct colors. — `fact:0266 lab_05_manual:p9`

**36. Hysteresis thresholding uses:**

- A) One threshold
- B) Two threshold values
- C) Three thresholds
- D) No threshold

Answer: **B) Two threshold values** ✅
Evidence (nli): Hysteresis thresholding (Canny) uses two thresholds: a high threshold to detect strong edges and a low threshold to link weak edges; example cv2.Canny(image, 100, 200). — `fact:0268 lab_05_manual:p10`

**37. In Canny edge detection, the high threshold is used to detect:**

- A) Weak edges
- B) Strong edges
- C) Noise
- D) Background

Answer: **B) Strong edges** ✅
Evidence (triple): high threshold | in Canny used to | detect strong edges — `triple:0042 lab_05_manual:p10`

**38. In Canny edge detection, the low threshold is used to:**

- A) Detect strong edges
- B) Link weak edges
- C) Remove noise
- D) Increase contrast

Answer: **B) Link weak edges** ✅
Evidence (triple): low threshold | in Canny used to | link weak edges — `triple:0043 lab_05_manual:p10`

**39. Hysteresis thresholding is typically used for:**

- A) Compression
- B) Edge detection
- C) Color correction
- D) Image resizing

Answer: **B) Edge detection** ✅
Evidence (nli): Canny edge detection steps: Gaussian smoothing reduces noise, gradient calculation finds magnitude and direction, non-maximum suppression, and hysteresis thresholding. — `fact:0316 lab_manual_06:p11`

**40. Edge detection detects boundaries by identifying:**

- A) Slow changes
- B) Rapid changes in intensity or color
- C) Constant regions
- D) Random noise

Answer: **B) Rapid changes in intensity or color** ✅
Evidence (nli): Edge detection segmentation detects boundaries by identifying rapid changes in intensity or color; useful for extracting contours and boundaries. — `fact:0269 lab_05_manual:p10`

**41. Edge detection is useful for:**

- A) Extracting object contours and boundaries
- B) Compressing images
- C) Enhancing colors
- D) Removing noise

Answer: **A) Extracting object contours and boundaries** ✅
Evidence (nli): Edge detection segmentation detects boundaries by identifying rapid changes in intensity or color; useful for extracting contours and boundaries. — `fact:0269 lab_05_manual:p10`

**42. Region growing starts with a:**

- A) Random pixel
- B) Seed pixel
- C) Threshold value
- D) Gradient

Answer: **B) Seed pixel** ✅
Evidence (nli): Region growing starts with a seed pixel and expands to neighboring pixels that are similar by a criterion such as intensity or color. — `fact:0270 lab_05_manual:p10`

**43. Region growing expands to neighboring pixels that are:**

- A) Different in intensity
- B) Similar in some criteria
- C) Random
- D) Edges

Answer: **A) Different in intensity** ❌ (key B: Similar in some criteria)
Evidence (nli): In region growing, a pixel whose intensity difference from the seed is below the threshold is added to the segmented region (mask set to 255) and its neighbours are pushed on the stack. — `fact:0525 lab_05_manual:p11`

**44. Region growing is effective for segmenting regions with:**

- A) Complex textures
- B) Uniform characteristics
- C) High noise
- D) Overlapping objects

Answer: **B) Uniform characteristics** ✅
Evidence (nli): Region growing is effective for segmenting regions with uniform characteristics. — `fact:0271 lab_05_manual:p10`

**45. In the region growing code, the seed point is initialized as:**

- A) (0,0)
- B) (10,10)
- C) (50,50)
- D) (100,100)

Answer: **B) (10,10)** ✅
Evidence (triple): region growing | seed point in the manual code | (10,10) — `triple:0072 lab_05_manual:p10`

**46. In the region growing code, the threshold value is:**

- A) 10
- B) 20
- C) 50
- D) 100

Answer: **C) 50** ✅
Evidence (triple): region growing | threshold value in the manual code | 50 — `triple:0073 lab_05_manual:p11`

**47. In region growing code, the mask is created with dtype:**

- A) np.float32
- B) np.uint8
- C) np.int16
- D) np.bool

Answer: **B) np.uint8** ✅
Evidence (triple): region growing | mask dtype in the manual code | np.uint8 — `triple:0074 lab_05_manual:p11`

**48. Region growing uses a stack for:**

- A) Sorting
- B) Pixel traversal
- C) Filtering
- D) Compression

Answer: **B) Pixel traversal** ✅
Evidence (nli): The region growing code pushes the 4-connected neighbours (x+1, y), (x-1, y), (x, y+1), (x, y-1) and uses a stack for pixel traversal. — `fact:0527 lab_05_manual:p11`

**49. If the intensity difference is below the threshold, the pixel is:**

- A) Ignored
- B) Added to the segmented region
- C) Marked as edge
- D) Removed

Answer: **B) Added to the segmented region** ✅
Evidence (nli): In region growing, a pixel whose intensity difference from the seed is below the threshold is added to the segmented region (mask set to 255) and its neighbours are pushed on the stack. — `fact:0525 lab_05_manual:p11`

**50. Region growing adds which neighboring pixels?**

- A) 8-connected
- B) 4-connected
- C) 16-connected
- D) All pixels

Answer: **B) 4-connected** ✅
Evidence (nli): The region growing code pushes the 4-connected neighbours (x+1, y), (x-1, y), (x, y+1), (x, y-1) and uses a stack for pixel traversal. — `fact:0527 lab_05_manual:p11`

**51. Watershed segmentation treats the image as a:**

- A) Binary image
- B) Topographic map
- C) Graph
- D) Histogram

Answer: **B) Topographic map** ✅
Evidence (nli): Watershed is suitable for segmenting touching or overlapping objects. — `fact:0274 lab_05_manual:p12`

**52. Watershed segmentation simulates:**

- A) Erosion
- B) Flooding
- C) Dilation
- D) Clustering

Answer: **B) Flooding** ✅
Evidence (nli): Watershed segmentation treats the image as a topographic map and simulates flooding to find segment boundaries. — `fact:0273 lab_05_manual:p12`

**53. Watershed segmentation is suitable for segmenting:**

- A) Isolated objects
- B) Touching or overlapping objects
- C) Binary images only
- D) Text

Answer: **B) Touching or overlapping objects** ✅
Evidence (nli): Watershed is suitable for segmenting touching or overlapping objects. — `fact:0274 lab_05_manual:p12`

**54. In watershed preprocessing, the input image is converted to:**

- A) RGB
- B) Grayscale
- C) HSV
- D) Binary

Answer: **B) Grayscale** ✅
Evidence (nli): Watershed preprocessing converts the input image to grayscale and may apply noise reduction or contrast enhancement. — `fact:0523 lab_05_manual:p12`

**55. Marker generation in watershed can be done by:**

- A) Manual placement
- B) Thresholding
- C) Distance transform
- D) All of the above

Answer: **D) All of the above** ✅
Evidence (nli): Watershed markers can be placed manually, created by thresholding, or created with a distance transform around object centers. — `fact:0276 lab_05_manual:p12`

**56. Gradient calculation in watershed uses:**

- A) Sobel or Scharr filters
- B) Gaussian blur
- C) Median filter
- D) Histogram equalization

Answer: **A) Sobel or Scharr filters** ✅
Evidence (nli): Watershed gradient calculation uses Sobel or Scharr filters to highlight object boundaries. — `fact:0524 lab_05_manual:p12`

**57. Marker labeling in watershed uses:**

- A) Binary values
- B) Different integer values
- C) Floating-point values
- D) Strings

Answer: **B) Different integer values** ✅
Evidence (nli): Watershed markers are labeled with different integer values; morphological erosion or dilation can refine the result. — `fact:0278 lab_05_manual:p12`

**58. In watershed, high gradient values represent:**

- A) Valleys
- B) Peaks/ridges
- C) Flat regions
- D) Noise

Answer: **B) Peaks/ridges** ✅
Evidence (nli): Watershed treats the gradient image as a topographic surface where high gradient values are peaks; basins fill from the labeled markers and meet at boundaries. — `fact:0277 lab_05_manual:p12`

**59. Boundaries where basins meet represent:**

- A) Noise
- B) Segmented regions
- C) Background
- D) Markers

Answer: **B) Segmented regions** ✅
Evidence (nli): The landscape example output segmented region map has segment 1 sky, segment 2 mountains and segment 3 water. — `fact:0514 lab_05_manual:p3`

**60. Watershed post-processing uses:**

- A) Morphological operations
- B) Fourier transform
- C) Histogram equalization
- D) Edge detection

Answer: **A) Morphological operations** ✅
Evidence (triple): watershed post-processing | watershed post-processing uses | morphological operations — `triple:0107 lab_05_manual:p12`

**61. In watershed code, threshold type used for markers is:**

- A) THRESH_BINARY
- B) THRESH_BINARY_INV + THRESH_OTSU
- C) THRESH_TRUNC
- D) THRESH_TOZERO

Answer: **B) THRESH_BINARY_INV + THRESH_OTSU** ✅
Evidence (triple): watershed code | threshold type used for markers | THRESH_BINARY_INV + THRESH_OTSU — `triple:0106 lab_05_manual:p13`

**62. Morphological opening removes:**

- A) Edges
- B) Noise
- C) Objects
- D) Colors

Answer: **B) Noise** ✅
Evidence (triple): opening | removes | noise — `triple:0061 opencv_docs:morphologyEx`

**63. In watershed code, sure_bg is created by:**

- A) Erosion
- B) Dilation
- C) Opening
- D) Closing

Answer: **B) Dilation** ✅
Evidence (triple): sure_bg | created by in watershed code | dilation — `triple:0075 lab_05_manual:p13`

**64. In watershed code, distanceTransform type is:**

- A) DIST_L1
- B) DIST_L2
- C) DIST_C
- D) DIST_L12

Answer: **B) DIST_L2** ✅
Evidence (nli): The watershed code gets the sure foreground by thresholding cv2.distanceTransform(opening, cv2.DIST_L2, 5) at 0.2 * max. — `fact:0280 lab_05_manual:p13`

**65. In watershed code, sure_fg threshold is set at:**

- A) 0.1 * max
- B) 0.2 * max
- C) 0.5 * max
- D) 0.8 * max

Answer: **B) 0.2 * max** ✅
Evidence (triple): sure_fg | threshold in watershed code | 0.2 * max — `triple:0076 lab_05_manual:p13`

**66. In watershed code, unknown region is calculated as:**

- A) add(sure_bg, sure_fg)
- B) subtract(sure_bg, sure_fg)
- C) multiply(sure_bg, sure_fg)
- D) divide(sure_bg, sure_fg)

Answer: **B) subtract(sure_bg, sure_fg)** ✅
Evidence (triple): unknown region | calculated as in watershed code | subtract(sure_bg, sure_fg) — `triple:0077 lab_05_manual:p13`

**67. In watershed code, connectedComponents returns:**

- A) Labels
- B) Markers
- C) Centers
- D) Contours

Answer: **A) Labels** ❌ (key B: Markers)
Evidence (nli): The watershed code creates markers with cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU). — `fact:0519 lab_05_manual:p13`

**68. After connectedComponents, markers are incremented by:**

- A) 0
- B) 1
- C) 2
- D) -1

Answer: **B) 1** ✅
Evidence (nli): The unknown region is sure background minus sure foreground (cv2.subtract); markers come from cv2.connectedComponents, +1 so background is 1, and unknown set to 0. — `fact:0281 lab_05_manual:p13`

**69. In watershed code, unknown region is set to:**

- A) 1
- B) 0
- C) -1
- D) 255

Answer: **B) 0** ✅
Evidence (triple): unknown region | set to in watershed code | 0 — `triple:0078 lab_05_manual:p13`

**70. In watershed, boundaries are marked as:**

- A) 0
- B) 1
- C) -1
- D) 255

Answer: **C) -1** ✅
Evidence (nli): The watershed code marks boundaries with image[markers == -1] = [255, 0, 0], commented as red. — `fact:0520 lab_05_manual:p13`

**71. In watershed code, boundaries are colored:**

- A) [0,0,255]
- B) [255,0,0] red
- C) [0,255,0] green
- D) [255,255,255] white

Answer: **B) [255,0,0] red** ✅
Evidence (nli): The watershed code marks boundaries with image[markers == -1] = [255, 0, 0], commented as red. — `fact:0520 lab_05_manual:p13`

**72. Clustering-based segmentation groups pixels based on:**

- A) File size
- B) Feature similarity
- C) Metadata
- D) Compression

Answer: **B) Feature similarity** ✅
Evidence (nli): Clustering-based segmentation groups pixels into clusters by feature similarity using K-Means, Mean-Shift or DBSCAN. — `fact:0283 lab_05_manual:p14`

**73. Which is a clustering technique?**

- A) K-Means
- B) Mean-Shift
- C) DBSCAN
- D) All of the above

Answer: **D) All of the above** ✅
Evidence (nli): K-means repeats assignment and update until convergence, when cluster assignments no longer change significantly; each cluster becomes a segment. — `fact:0287 lab_05_manual:p14`

**74. Feature extraction for color images can use:**

- A) RGB
- B) LAB
- C) HSV
- D) All of the above

Answer: **D) All of the above** ✅
Evidence (nli): In the Lab 05 landscape figure, preprocessing and feature extraction uses pixel values, edge detection, texture analysis and color similarity. — `fact:0512 lab_05_manual:p3`

**75. Cluster initialization decides:**

- A) Number of clusters K
- B) Number of pixels
- C) Image size
- D) Threshold

Answer: **A) Number of clusters K** ✅
Evidence (nli): K-means initialization chooses the number of clusters K and initializes K centroids randomly or by a heuristic. — `fact:0285 lab_05_manual:p14`

**76. In K-Means, the assignment step assigns each pixel to:**

- A) Random cluster
- B) Nearest cluster centroid
- C) Farthest cluster
- D) All clusters

Answer: **B) Nearest cluster centroid** ✅
Evidence (nli): K-means assignment step: assign each pixel to the nearest centroid by Euclidean distance; update step: recompute each centroid as the mean of its pixels. — `fact:0286 lab_05_manual:p14`

**77. In K-Means, the update step recalculates centroids as:**

- A) Median of pixels
- B) Mean of pixels in each cluster
- C) Mode of pixels
- D) Random values

Answer: **B) Mean of pixels in each cluster** ✅
Evidence (triple): update step | recalculates centroids as | mean of pixels in each cluster — `triple:0109 lab_05_manual:p14`

**78. K-Means convergence is reached when:**

- A) Centroids are random
- B) Cluster assignments no longer change significantly
- C) All pixels are in one cluster
- D) Image is compressed

Answer: **B) Cluster assignments no longer change significantly** ✅
Evidence (nli): K-means repeats assignment and update until convergence, when cluster assignments no longer change significantly; each cluster becomes a segment. — `fact:0287 lab_05_manual:p14`

**79. After convergence, each cluster represents:**

- A) Noise
- B) A segment or region
- C) An edge
- D) A pixel

Answer: **B) A segment or region** ✅
Evidence (nli): K-means repeats assignment and update until convergence, when cluster assignments no longer change significantly; each cluster becomes a segment. — `fact:0287 lab_05_manual:p14`

**80. Post-processing in clustering can involve:**

- A) Merging or splitting clusters
- B) Compression
- C) Encryption
- D) Resizing

Answer: **A) Merging or splitting clusters** ✅
Evidence (triple): clustering | post-processing can involve | merging or splitting clusters — `triple:0108 lab_05_manual:p14`

**81. In K-Means code, pixel_values is reshaped to:**

- A) (-1, 2)
- B) (-1, 3)
- C) (-1, 4)
- D) (-1, 1)

Answer: **B) (-1, 3)** ✅
Evidence (triple): pixel_values | reshaped to in K-Means code | (-1, 3) — `triple:0081 lab_05_manual:p15`

**82. In K-Means code, pixel_values is converted to:**

- A) np.uint8
- B) np.float32
- C) np.int16
- D) np.bool

Answer: **B) np.float32** ✅
Evidence (triple): pixel_values | converted to in K-Means code | np.float32 — `triple:0082 lab_05_manual:p15`

**83. In K-Means code, criteria is:**

- A) TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 100, 0.2
- B) TERM_CRITERIA_EPS, 50, 0.1
- C) TERM_CRITERIA_MAX_ITER, 200, 0.5
- D) None

Answer: **A) TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 100, 0.2** ✅
Evidence (nli): K-means criteria combine cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER (100 iterations, epsilon 0.2); centers are converted back with np.uint8. — `fact:0289 lab_05_manual:p15`

**84. In K-Means code, K value is:**

- A) 2
- B) 3
- C) 4
- D) 5

Answer: **C) 4** ✅
Evidence (triple): K | in K-Means code | 4 — `triple:0080 lab_05_manual:p15`

**85. In K-Means code, flags used is:**

- A) KMEANS_PP_CENTERS
- B) KMEANS_RANDOM_CENTERS
- C) KMEANS_USE_INITIAL_LABELS
- D) None

Answer: **B) KMEANS_RANDOM_CENTERS** ✅
Evidence (nli): The K-means code reshapes the image to (-1, 3), converts to np.float32, and calls cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) with K = 4. — `fact:0288 lab_05_manual:p15`

**86. In K-Means code, centers are converted to:**

- A) np.float32
- B) np.uint8
- C) np.int16
- D) np.bool

Answer: **B) np.uint8** ✅
Evidence (triple): centers | converted to in K-Means code | np.uint8 — `triple:0083 lab_05_manual:p15`

**87. Deep learning-based segmentation uses:**

- A) CNNs
- B) RNNs
- C) GANs
- D) SVMs

Answer: **A) CNNs** ✅
Evidence (nli): Deep learning-based segmentation uses CNNs for semantic and instance segmentation; suitable for complex tasks and large datasets. — `fact:0291 lab_05_manual:p15`

**88. Semantic segmentation means:**

- A) Different instances of same class separated
- B) Same class gets same label
- C) Random labels
- D) No labels

Answer: **B) Same class gets same label** ✅
Evidence (triple): semantic segmentation | means | same class gets same label — `triple:0067 cv_fundamentals:segmentation`

**89. Instance segmentation means:**

- A) Same class gets same label
- B) Different instances of same class are separated
- C) No labels
- D) Binary labels

Answer: **B) Different instances of same class are separated** ✅
Evidence (triple): instance segmentation | means | different instances of same class are separated — `triple:0068 cv_fundamentals:segmentation`

**90. Deep learning segmentation is suitable for:**

- A) Simple tasks
- B) Complex tasks and large datasets
- C) Small datasets only
- D) Binary images only

Answer: **B) Complex tasks and large datasets** ✅
Evidence (nli): Deep learning-based segmentation uses CNNs for semantic and instance segmentation; suitable for complex tasks and large datasets. — `fact:0291 lab_05_manual:p15`

**91. Active contour models are also called:**

- A) Snakes
- B) Birds
- C) Worms
- D) Fish

Answer: **A) Snakes** ✅
Evidence (nli): Active contour models (snakes) are deformable models that evolve to object boundaries using image gradients and constraints; used for object tracking and boundary refinement. — `fact:0293 lab_05_manual:p16`

**92. Active contour models evolve based on:**

- A) Random motion
- B) Image gradients and constraints
- C) Histogram
- D) Fourier transform

Answer: **B) Image gradients and constraints** ✅
Evidence (nli): Active contour models (snakes) are deformable models that evolve to object boundaries using image gradients and constraints; used for object tracking and boundary refinement. — `fact:0293 lab_05_manual:p16`

**93. Active contour models are effective for:**

- A) Object tracking and boundary refinement
- B) Compression
- C) Color correction
- D) Noise removal

Answer: **A) Object tracking and boundary refinement** ✅
Evidence (nli): Edge detection segmentation detects boundaries by identifying rapid changes in intensity or color; useful for extracting contours and boundaries. — `fact:0269 lab_05_manual:p10`

**94. cv2.imread() flag for grayscale is:**

- A) IMREAD_COLOR
- B) IMREAD_GRAYSCALE
- C) IMREAD_UNCHANGED
- D) IMREAD_ANYCOLOR

Answer: **B) IMREAD_GRAYSCALE** ✅
Evidence (triple): imread | flag for grayscale | IMREAD_GRAYSCALE — `triple:0126 lab_01_manual:p24`

**95. cv2.threshold() returns:**

- A) Only binary mask
- B) retval and dst
- C) Only retval
- D) None

Answer: **B) retval and dst** ✅
Evidence (triple): cv2.threshold | returns | retval and dst — `triple:0217 opencv_introspection:opencv-4.13.0`

**96. Otsu thresholding is combined with:**

- A) THRESH_TRUNC
- B) THRESH_BINARY + THRESH_OTSU
- C) THRESH_TOZERO
- D) THRESH_BINARY_INV only

Answer: **B) THRESH_BINARY + THRESH_OTSU** ✅
Evidence (nli): Otsu thresholding is requested by combining flags, e.g. cv2.THRESH_BINARY + cv2.THRESH_OTSU or cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU; the thresh argument is ignored and retval is the computed threshold. — `fact:0373 opencv_docs:threshold`

**97. cv2.cvtColor() code for BGR to HSV is:**

- A) COLOR_BGR2GRAY
- B) COLOR_BGR2HSV
- C) COLOR_BGR2RGB
- D) COLOR_BGR2LAB

Answer: **B) COLOR_BGR2HSV** ✅
Evidence (nli): Color thresholding converts to HSV with cv2.cvtColor(image, cv2.COLOR_BGR2HSV) and builds a mask with cv2.inRange(hsv_image, lower_bound, upper_bound). — `fact:0267 lab_05_manual:p9`

**98. cv2.inRange() parameters are:**

- A) src, lowerb, upperb
- B) src, threshold
- C) src, kernel
- D) src, markers

Answer: **A) src, lowerb, upperb** ✅
Evidence (triple): cv2.inRange | parameters | src, lowerb, upperb — `triple:0063 opencv_docs:inRange`

**99. cv2.Canny() parameters are:**

- A) image, threshold1, threshold2
- B) image, K
- C) image, markers
- D) image, kernel

Answer: **A) image, threshold1, threshold2** ✅
Evidence (triple): cv2.Canny | parameters | image, threshold1, threshold2 — `triple:0064 opencv_docs:Canny`

**100. cv2.kmeans() returns:**

- A) compactness, labels, centers
- B) retval, dst
- C) markers
- D) edges

Answer: **A) compactness, labels, centers** ✅
Evidence (triple): cv2.kmeans | returns | compactness, labels, centers — `triple:0218 opencv_introspection:opencv-4.13.0`

**101. cv2.watershed() marks boundaries as:**

- A) 0
- B) 1
- C) -1
- D) 255

Answer: **C) -1** ✅
Evidence (nli): cv2.watershed(image, markers) needs an 8-bit 3-channel image and int32 markers; it modifies markers in place and marks boundary pixels with -1. — `fact:0384 opencv_docs:watershed`

**102. cv2.adaptiveThreshold() parameter for neighborhood size is:**

- A) C
- B) blockSize
- C) maxValue
- D) thresholdType

Answer: **B) blockSize** ✅
Evidence (nli): cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C): blockSize is the odd neighbourhood size used to compute the local threshold. — `fact:0375 opencv_docs:adaptiveThreshold`

**103. cv2.morphologyEx() with MORPH_OPEN performs:**

- A) Dilation then erosion
- B) Erosion then dilation
- C) Only erosion
- D) Only dilation

Answer: **B) Erosion then dilation** ✅
Evidence (nli): Morphological opening (MORPH_OPEN) is erosion followed by dilation; it removes small noise (white specks) while keeping object shape. — `fact:0378 opencv_docs:morphologyEx`
