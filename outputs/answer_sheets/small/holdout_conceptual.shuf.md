# Answer sheet: holdout_conceptual.shuf.txt

Graded 25/40 correct. Wrong: Q3, Q4, Q8, Q9, Q12, Q18, Q19, Q20, Q22, Q24, Q26, Q29, Q31, Q32, Q40

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. Why is an image usually smoothed with a Gaussian before running Canny?**

- A) To make the edges thicker
- B) To convert the image to grayscale
- C) To suppress noise that would otherwise produce false edges
- D) To increase the number of color channels

Answer: **C) To suppress noise that would otherwise produce false edges** ✅
Evidence (nli): Gaussian smoothing in Canny reduces noise to prevent false edges; non-maximum suppression thins thick edges to one-pixel-wide lines. — `fact:0319 lab_manual_06:p12`

**2. A photo of a page has a shadow across one corner. Which thresholding approach handles it best?**

- A) Adaptive thresholding that computes a local threshold per neighborhood
- B) Inverting the image
- C) Histogram of oriented gradients
- D) A single global threshold of 127

Answer: **A) Adaptive thresholding that computes a local threshold per neighborhood** ✅
Evidence (nli): Adaptive thresholding uses different threshold values for different regions, adapting to local intensity variations; ideal for varying lighting conditions. — `fact:0261 lab_05_manual:p8`

**3. Two coins are touching each other in a binary mask. Which method is designed to split them into separate objects?**

- A) Marker-controlled watershed
- B) Global thresholding
- C) Histogram equalization
- D) Gaussian blur

Answer: **B) Global thresholding** ❌ (key A: Marker-controlled watershed)
Evidence (nli): The global threshold example is cv2.threshold(image, 128, 255, cv2.THRESH_BINARY). — `fact:0260 lab_05_manual:p8`

**4. You need a mask of all orange objects in a colorful image. Which pipeline is most appropriate?**

- A) Use cv2.HoughCircles
- B) Apply cv2.equalizeHist
- C) Convert to HSV and apply cv2.inRange with lower and upper bounds
- D) Apply cv2.Canny to the color image

Answer: **A) Use cv2.HoughCircles** ❌ (key C: Convert to HSV and apply cv2.inRange with lower and upper bounds)
Evidence (nli): cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius) implements the cv2.HOUGH_GRADIENT method (and HOUGH_GRADIENT_ALT). — `fact:0391 opencv_docs:HoughCircles`

**5. An 8-bit result of cv2.Sobel loses information because:**

- A) The image becomes RGB
- B) Negative gradient values are clipped to zero
- C) The kernel becomes 5x5
- D) Sobel only works on float images

Answer: **B) Negative gradient values are clipped to zero** ✅
Evidence (nli): A signed float output depth such as cv2.CV_64F keeps negative gradient values that an 8-bit cv2.CV_8U output would clip to 0. — `fact:0363 opencv_docs:Sobel`

**6. After rotating an image by 45 degrees with the same canvas size, the corners are cut off because:**

- A) warpAffine converts the image to grayscale
- B) The rotation is done around the top-left corner only
- C) The output canvas keeps the original width and height
- D) getRotationMatrix2D always scales by 2

Answer: **C) The output canvas keeps the original width and height** ✅
Evidence (nli): In the image coordinate convention the origin (0,0) is the top-left corner; M is the total rows (height) and N the total columns (width). — `fact:0164 lab_01_manual:p10`

**7. Which property is preserved by an affine transformation but NOT necessarily by a projective transformation?**

- A) Straightness of lines
- B) Pixel intensity
- C) Number of pixels
- D) Parallelism of lines

Answer: **D) Parallelism of lines** ✅
Evidence (nli): Affine transformations always preserve parallelism; projective transformations do not. — `fact:0238 lab_03_manual:p18`

**8. Why does a 2x2 matrix need to be extended to 3x3 to represent translation?**

- A) Translation changes pixel colors
- B) Multiplying the origin by any 2x2 matrix always gives the origin
- C) 3x3 matrices are faster to store
- D) 2x2 matrices cannot represent rotation

Answer: **C) 3x3 matrices are faster to store** ❌ (key B: Multiplying the origin by any 2x2 matrix always gives the origin)
Evidence (nli): Sobel and Scharr operators detect edges by emphasizing rapid changes in pixel values. — `fact:0092 lab_04_manual:p17`

**9. A photographed rectangular document appears as a trapezoid. Which transformation restores the rectangle?**

- A) Pure rotation
- B) Uniform scaling
- C) Projective (homography)
- D) Gamma correction

Answer: **A) Pure rotation** ❌ (key C: Projective (homography))
Evidence (nli): A pure rotation must change all four matrix values together using sine and cosine so both axes stay perpendicular (90 degrees apart). — `fact:0221 lab_03_manual:p11`

**10. Which statement about Otsu's method is correct?**

- A) It requires the user to provide the threshold
- B) It is a color-space conversion
- C) It works best on images with a single intensity peak
- D) It assumes a roughly bimodal histogram and picks the threshold automatically

Answer: **D) It assumes a roughly bimodal histogram and picks the threshold automatically** ✅
Evidence (nli): A bimodal histogram has two separated peaks (object and background), the case where a single global or Otsu threshold works well. — `fact:0415 cv_fundamentals:histogram`

**11. In K-means color segmentation with K = 3, the output image contains at most:**

- A) 1 color
- B) 3 distinct colors
- C) 256 distinct colors
- D) 9 distinct colors

Answer: **B) 3 distinct colors** ✅
Evidence (nli): The K-means code reshapes the image to (-1, 3), converts to np.float32, and calls cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) with K = 4. — `fact:0288 lab_05_manual:p15`

**12. Why must pixel data be converted to float32 before cv2.kmeans?**

- A) It reduces the number of clusters
- B) float32 removes noise
- C) It converts BGR to RGB
- D) cv2.kmeans requires floating-point samples

Answer: **C) It converts BGR to RGB** ❌ (key D: cv2.kmeans requires floating-point samples)
Evidence (nli): cv2.kmeans(data, K, bestLabels, criteria, attempts, flags): data must be float32 with one row per sample; attempts is the number of runs with different initial centers. — `fact:0386 opencv_docs:kmeans`

**13. Morphological opening applied to a binary mask mainly:**

- A) Detects lines
- B) Converts the mask to color
- C) Removes small isolated white specks
- D) Fills small black holes inside objects

Answer: **C) Removes small isolated white specks** ✅
Evidence (nli): The Hough transform was introduced by Paul Hough in 1962 for detecting lines in binary images, later extended to circles and ellipses. — `fact:0327 lab_manual_06:p17`

**14. Morphological closing applied to a binary mask mainly:**

- A) Removes small white specks
- B) Finds keypoints
- C) Fills small holes and gaps in objects
- D) Computes gradients

Answer: **C) Fills small holes and gaps in objects** ✅
Evidence (nli): Erosion shrinks bright foreground regions and removes small specks; dilation grows bright regions and fills small gaps. — `fact:0381 opencv_docs:erode`

**15. In marker-based watershed, what does the distance transform help find?**

- A) The color of each object
- B) Straight lines
- C) The image histogram
- D) The sure foreground near object centers

Answer: **D) The sure foreground near object centers** ✅
Evidence (nli): The watershed code gets the sure foreground by thresholding cv2.distanceTransform(opening, cv2.DIST_L2, 5) at 0.2 * max. — `fact:0280 lab_05_manual:p13`

**16. Which edge detector produces thin, one-pixel-wide, connected edges?**

- A) Canny
- B) Plain Sobel magnitude
- C) Box blur
- D) Global threshold

Answer: **A) Canny** ✅
Evidence (nli): Gaussian smoothing in Canny reduces noise to prevent false edges; non-maximum suppression thins thick edges to one-pixel-wide lines. — `fact:0319 lab_manual_06:p12`

**17. Edges in a Laplacian-of-Gaussian response are located at:**

- A) Zero-crossings of the response
- B) Pixels with value 255
- C) The brightest pixels only
- D) The image corners

Answer: **A) Zero-crossings of the response** ✅
Evidence (nli): LoG stands for Laplacian of Gaussian; it combines Gaussian smoothing and Laplacian edge detection and highlights zero-crossings of the second derivative. — `fact:0105 lab_04_manual:p20`

**18. Why does SIFT assign an orientation to each keypoint?**

- A) To count the number of keypoints
- B) So the descriptor can be computed relative to it, making matching rotation invariant
- C) To convert the image to grayscale
- D) To speed up Gaussian blur

Answer: **A) To count the number of keypoints** ❌ (key B: So the descriptor can be computed relative to it, making matching rotation invariant)
Evidence (nli): len(keypoints) gives the number of keypoints and descriptors.shape is (number of keypoints, 128). — `fact:0349 lab_manual_06:p23`

**19. Why is RANSAC used after matching keypoints between two images?**

- A) To convert colors
- B) To reject wrong matches that do not fit a common geometric transform
- C) To compute the SIFT descriptor
- D) To blur the images

Answer: **C) To compute the SIFT descriptor** ❌ (key B: To reject wrong matches that do not fit a common geometric transform)
Evidence (nli): RANSAC keeps only geometrically consistent matches (inliers). — `fact:0345 lab_manual_06:p22`

**20. Lowe's ratio test keeps a match when:**

- A) The keypoint is at the image border
- B) The two images have the same size
- C) The descriptor has 64 values
- D) The best match distance is clearly smaller than the second-best distance

Answer: **B) The two images have the same size** ❌ (key D: The best match distance is clearly smaller than the second-best distance)
Evidence (nli): Images must have the same dimensions before they can be added or blended; cv2.resize makes them match. — `fact:0199 lab_01_manual:p23`

**21. Which descriptor should be matched with Hamming distance?**

- A) ORB binary descriptors
- B) HOG descriptors
- C) SIFT float descriptors
- D) Color histograms

Answer: **A) ORB binary descriptors** ✅
Evidence (nli): cv2.BFMatcher(normType, crossCheck): use NORM_L2 for float descriptors like SIFT and NORM_HAMMING for binary descriptors like ORB. — `fact:0398 opencv_docs:BFMatcher`

**22. In the Hough line transform, a peak in the accumulator corresponds to:**

- A) A single noisy pixel
- B) Many edge pixels voting for the same line parameters
- C) The brightest color
- D) The image center

Answer: **D) The image center** ❌ (key B: Many edge pixels voting for the same line parameters)
Evidence (nli): Hough peak detection: peaks in the parameter space (accumulator) represent lines; back-transformation maps them to image positions. — `fact:0331 lab_manual_06:p17`

**23. Why do practical Hough line implementations use rho and theta instead of slope and intercept?**

- A) Intercepts are always zero
- B) rho and theta use less memory for color images
- C) Slope cannot be negative
- D) Vertical lines have infinite slope

Answer: **D) Vertical lines have infinite slope** ✅
Evidence (nli): In the slope-intercept Hough form a line y = mx + b maps to a point (m, b); vertical lines have infinite slope, so the polar (rho, theta) form is used in practice. — `fact:0413 cv_fundamentals:hough`

**24. Increasing param2 in cv2.HoughCircles usually:**

- A) Changes the image color
- B) Detects fewer, more reliable circles
- C) Increases the image size
- D) Detects more false circles

Answer: **D) Detects more false circles** ❌ (key B: Detects fewer, more reliable circles)
Evidence (nli): In cv2.HoughCircles with HOUGH_GRADIENT, param1 is the higher threshold passed to the Canny edge detector and param2 is the accumulator threshold for circle centers (smaller gives more false circles). — `fact:0393 opencv_docs:HoughCircles`

**25. A valid (no padding) convolution of a 10x10 image with a 3x3 kernel gives an output of size:**

- A) 12x12
- B) 8x8
- C) 10x10
- D) 3x3

Answer: **B) 8x8** ✅
Evidence (nli): HOG cell division: the image is divided into small non-overlapping cells, typical cell size 8x8 pixels. — `fact:0025 lab_04_manual:p6`

**26. A same convolution of a 10x10 image with a 3x3 kernel gives an output of size:**

- A) 7x7
- B) 8x8
- C) 12x12
- D) 10x10

Answer: **B) 8x8** ❌ (key D: 10x10)
Evidence (nli): HOG cell division: the image is divided into small non-overlapping cells, typical cell size 8x8 pixels. — `fact:0025 lab_04_manual:p6`

**27. Applying a 3x3 box filter replaces each pixel with:**

- A) The mean of its 3x3 neighborhood
- B) The median of its neighborhood
- C) The center pixel unchanged
- D) The maximum of its neighborhood

Answer: **A) The mean of its 3x3 neighborhood** ✅
Evidence (nli): Box blur replaces each pixel with the average of its neighboring pixels within a square kernel; useful for noise reduction and smoothing. — `fact:0088 lab_04_manual:p16`

**28. Which filter is best for removing salt-and-pepper noise while keeping edges fairly sharp?**

- A) Median filter
- B) Box filter
- C) Embossing kernel
- D) Sobel filter

Answer: **A) Median filter** ✅
Evidence (nli): Filtering applies a filter or kernel, a small matrix of numbers, to an input image. — `fact:0084 lab_04_manual:p16`

**29. Which filter smooths flat regions while preserving strong edges by also weighting intensity similarity?**

- A) Laplacian
- B) Box filter
- C) Sobel
- D) Bilateral filter

Answer: **A) Laplacian** ❌ (key D: Bilateral filter)
Evidence (nli): LoG stands for Laplacian of Gaussian; it combines Gaussian smoothing and Laplacian edge detection and highlights zero-crossings of the second derivative. — `fact:0105 lab_04_manual:p20`

**30. HOG block normalization mainly provides robustness to:**

- A) Changes in image file format
- B) Changes in lighting and contrast
- C) Changes in color space
- D) Image compression

Answer: **B) Changes in lighting and contrast** ✅
Evidence (nli): HOG block normalization groups neighboring cells into blocks (typical block 2x2 cells) and normalizes them to reduce the effects of lighting variations. — `fact:0027 lab_04_manual:p7`

**31. Uniform LBP patterns are useful because they:**

- A) Detect circles
- B) Increase the image resolution
- C) Convert texture to color
- D) Reduce the number of histogram bins while keeping most texture information

Answer: **A) Detect circles** ❌ (key D: Reduce the number of histogram bins while keeping most texture information)
Evidence (nli): In the LBP code method='uniform' uses uniform patterns, which reduces the number of patterns (n_points + 2 histogram bins). — `fact:0043 lab_04_manual:p9`

**32. An image of a bright uniform white wall has high texture energy but:**

- A) Zero texture contrast
- B) No histogram
- C) High texture contrast
- D) Negative energy

Answer: **C) High texture contrast** ❌ (key A: Zero texture contrast)
Evidence (nli): Higher texture contrast values indicate greater variation in texture. — `fact:0074 lab_04_manual:p13`

**33. Why is a color histogram considered a global feature?**

- A) It is learned by a CNN
- B) It is computed only at keypoints
- C) It is computed over the entire image
- D) It depends on corner positions

Answer: **C) It is computed over the entire image** ✅
Evidence (nli): Histogram of Color (a color histogram) is an example of a global feature because it is computed over the entire image. — `fact:0047 lab_04_manual:p5`

**34. Which operation changes what a pixel looks like but not where it is?**

- A) Rotation
- B) Shearing
- C) Gamma correction
- D) Translation

Answer: **C) Gamma correction** ✅
Evidence (nli): LBP captures local patterns by comparing the intensity of a pixel with its neighboring pixels. — `fact:0033 lab_04_manual:p7`

**35. cv2.add(200, 100) on uint8 images gives 255 because:**

- A) NumPy wraps around modulo 256
- B) The images are converted to float
- C) OpenCV uses saturation arithmetic
- D) The result is divided by 2

Answer: **C) OpenCV uses saturation arithmetic** ✅
Evidence (nli): cv2.add performs saturation arithmetic: sums above 255 are capped at 255 instead of wrapping around, so the result looks brighter. — `fact:0200 lab_01_manual:p23`

**36. Histogram equalization improves a low-contrast image by:**

- A) Converting it to HSV
- B) Blurring it
- C) Spreading the clustered intensity values over the full 0-255 range
- D) Detecting its edges

Answer: **C) Spreading the clustered intensity values over the full 0-255 range** ✅
Evidence (nli): cv2.equalizeHist spreads clumped pixel intensities across the full 0-255 range, improving global contrast of dark or washed-out images. — `fact:0202 lab_01_manual:p24`

**37. Which segmentation approach starts from a seed pixel and adds similar neighbors?**

- A) Region growing
- B) Watershed
- C) K-means clustering
- D) Global thresholding

Answer: **A) Region growing** ✅
Evidence (nli): In region growing, a pixel whose intensity difference from the seed is below the threshold is added to the segmented region (mask set to 255) and its neighbours are pushed on the stack. — `fact:0525 lab_05_manual:p11`

**38. Instance segmentation differs from semantic segmentation because it:**

- A) Does not assign labels
- B) Uses only grayscale images
- C) Only works on videos
- D) Separates individual objects of the same class

Answer: **D) Separates individual objects of the same class** ✅
Evidence (nli): Semantic segmentation gives every pixel of the same class the same label; instance segmentation also separates different instances (objects) of the same class. — `fact:0403 cv_fundamentals:segmentation`

**39. Why do practitioners convert an OpenCV image with COLOR_BGR2RGB before plt.imshow?**

- A) Matplotlib interprets three-channel arrays as RGB, OpenCV stores BGR
- B) Matplotlib cannot display grayscale
- C) It reduces file size
- D) It equalizes the histogram

Answer: **A) Matplotlib interprets three-channel arrays as RGB, OpenCV stores BGR** ✅
Evidence (nli): Before plotting with Matplotlib, convert OpenCV's default BGR (Blue, Green, Red) to RGB with cv2.cvtColor(image, cv2.COLOR_BGR2RGB). — `fact:0179 lab_01_manual:p14`

**40. Which statement about the Difference of Gaussians in SIFT is correct?**

- A) It approximates the Laplacian of Gaussian across scales
- B) It removes keypoints at all scales
- C) It is the RANSAC inlier test
- D) It is a color conversion

Answer: **D) It is a color conversion** ❌ (key A: It approximates the Laplacian of Gaussian across scales)
Evidence (nli): The integer value of cv2.COLOR_BGR2HSV (color conversion code) is 40. — `fact:0426 opencv_introspection:opencv-4.13.0`
