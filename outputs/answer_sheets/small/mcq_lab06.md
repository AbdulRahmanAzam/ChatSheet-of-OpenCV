# Answer sheet: mcq_lab06.txt

Graded 96/100 correct. Wrong: Q17, Q22, Q44, Q86

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. What does CWT stand for?**

- A) Continuous Wavelet Transform
- B) Discrete Wavelet Transform
- C) Wavelet Packet Transform
- D) Inverse Wavelet Transform

Answer: **A) Continuous Wavelet Transform** ✅
Evidence (nli): The CWT uses continuous wavelet functions (psi) that are dilated (scaled) and translated across the signal. — `fact:0298 lab_manual_06:p3`

**2. Wavelet transformation provides analysis in which two domains?**

- A) Time and frequency
- B) Space and color
- C) Magnitude and phase
- D) Gradient and Laplacian

Answer: **A) Time and frequency** ✅
Evidence (nli): The Fourier transform gives frequency content only; wavelets give joint time (space) and frequency localization. — `fact:0416 cv_fundamentals:wavelets`

**3. Which type of signals is wavelet transformation especially useful for?**

- A) Stationary signals
- B) Non-stationary signals
- C) Periodic signals only
- D) Binary signals only

Answer: **B) Non-stationary signals** ✅
Evidence (nli): Wavelet transforms provide time-frequency analysis, particularly useful for non-stationary signals. — `fact:0296 lab_manual_06:p3`

**4. In CWT, the wavelet function is:**

- A) Only translated
- B) Only dilated
- C) Dilated and translated
- D) Neither dilated nor translated

Answer: **C) Dilated and translated** ✅
Evidence (nli): The CWT uses continuous wavelet functions (psi) that are dilated (scaled) and translated across the signal. — `fact:0298 lab_manual_06:p3`

**5. Which wavelet is mentioned for analyzing oscillatory components?**

- A) Haar
- B) Daubechies
- C) Morlet
- D) Symlet

Answer: **C) Morlet** ✅
Evidence (nli): The Morlet wavelet is used with the CWT for analyzing oscillatory components in signals. — `fact:0299 lab_manual_06:p3`

**6. Which wavelet families are used in DWT?**

- A) Morlet, Mexican Hat, Shannon
- B) Daubechies, Haar, Symlet
- C) Gabor, Fourier, Laplace
- D) Sobel, Canny, LoG

Answer: **B) Daubechies, Haar, Symlet** ✅
Evidence (nli): DWT wavelet families include Daubechies, Haar and Symlet wavelets. — `fact:0302 lab_manual_06:p3`

**7. WPT is an extension of:**

- A) CWT
- B) DWT
- C) Inverse Wavelet Transform
- D) Fourier Transform

Answer: **B) DWT** ✅
Evidence (triple): WPT | extension of | DWT — `triple:0096 lab_manual_06:p4`

**8. The Inverse Wavelet Transform is crucial for:**

- A) Edge detection
- B) Image reconstruction
- C) Hough voting
- D) SIFT matching

Answer: **B) Image reconstruction** ✅
Evidence (triple): Inverse Wavelet Transform | crucial for | image reconstruction — `triple:0098 lab_manual_06:p4`

**9. Which application is NOT mentioned for DWT?**

- A) Image compression
- B) Image denoising
- C) Feature extraction
- D) Line detection

Answer: **D) Line detection** ✅
Evidence (nli): For line detection the manual represents a line as y = mx + b, with slope m and y-intercept b as parameter space axes. — `fact:0329 lab_manual_06:p17`

**10. WPT is useful for analyzing signals with:**

- A) Simple frequency content
- B) Complex frequency content
- C) No frequency content
- D) Only binary content

Answer: **B) Complex frequency content** ✅
Evidence (nli): WPT is used for signals with complex frequency content and for feature extraction in machine learning. — `fact:0305 lab_manual_06:p4`

**11. Boundary detection is also known as:**

- A) Edge detection
- B) Corner detection
- C) Blob detection
- D) Circle detection

Answer: **A) Edge detection** ✅
Evidence (nli): Boundary detection, often called edge detection, identifies object boundaries: significant changes in intensity or color. — `fact:0307 lab_manual_06:p5`

**12. Edges represent significant changes in:**

- A) Intensity or color
- B) Only color
- C) Only intensity
- D) Only texture

Answer: **A) Intensity or color** ✅
Evidence (nli): Boundary detection, often called edge detection, identifies object boundaries: significant changes in intensity or color. — `fact:0307 lab_manual_06:p5`

**13. Which task is NOT mentioned as using edges?**

- A) Object recognition
- B) Image segmentation
- C) Feature extraction
- D) Audio compression

Answer: **D) Audio compression** ✅
Evidence (nli): Applications of feature extraction: object recognition, face recognition, gesture recognition, medical imaging and autonomous vehicles. — `fact:0014 lab_04_manual:p5`

**14. Sobel edge detection is based on:**

- A) Gradient
- B) Laplacian
- C) Zero-crossings
- D) Hough voting

Answer: **A) Gradient** ✅
Evidence (nli): Gradient-based edge detection calculates the gradient (rate of change) of pixel intensities. — `fact:0101 lab_04_manual:p20`

**15. How many convolution kernels does Sobel use?**

- A) One
- B) Two
- C) Three
- D) Four

Answer: **B) Two** ✅
Evidence (triple): Sobel | number of convolution kernels | two — `triple:0112 lab_manual_06:p6`

**16. What is the size of Sobel kernels?**

- A) 2×2
- B) 3×3
- C) 5×5
- D) 7×7

Answer: **B) 3×3** ✅
Evidence (triple): Sobel kernels | kernel size | 3x3 — `triple:0113 lab_manual_06:p6`

**17. Which Sobel kernel detects vertical edges?**

- A) Gx
- B) Gy
- C) Gz
- D) G

Answer: **B) Gy** ❌ (key A: Gx)
Evidence (nli): cv2.Sobel with dx=1, dy=0 computes the horizontal gradient Gx (derivative along x), which responds to vertical edges. — `fact:0360 opencv_docs:Sobel`

**18. Which Sobel kernel detects horizontal edges?**

- A) Gx
- B) Gy
- C) Gz
- D) G

Answer: **B) Gy** ✅
Evidence (nli): cv2.Sobel with dx=0, dy=1 computes the vertical gradient Gy (derivative along y), which responds to horizontal edges. — `fact:0361 opencv_docs:Sobel`

**19. The Sobel gradient magnitude is calculated as:**

- A) Gx + Gy
- B) Gx² + Gy²
- C) sqrt(Gx² + Gy²)
- D) Abs(Gx) + Abs(Gy)

Answer: **C) sqrt(Gx² + Gy²)** ✅
Evidence (nli): Sobel total gradient magnitude G = sqrt(Gx^2 + Gy^2) combines horizontal and vertical changes. — `fact:0314 lab_manual_06:p8`

**20. Why is CV_64F used in Sobel?**

- A) To preserve negative values
- B) To increase brightness
- C) To reduce noise
- D) To detect circles

Answer: **A) To preserve negative values** ✅
Evidence (nli): A signed float output depth such as cv2.CV_64F keeps negative gradient values that an 8-bit cv2.CV_8U output would clip to 0. — `fact:0363 opencv_docs:Sobel`

**21. Why are grayscale images often used in computer vision tasks?**

- A) To simplify intensity analysis
- B) To increase color information
- C) To detect circles
- D) To perform Hough voting

Answer: **A) To simplify intensity analysis** ✅
Evidence (nli): LBP operates on grayscale images and defines a circular neighborhood around each pixel. — `fact:0035 lab_04_manual:p8`

**22. What is the correct order of Canny edge detection steps?**

- A) Gradient → Gaussian smoothing → NMS → Hysteresis
- B) Gaussian smoothing → Gradient → NMS → Hysteresis
- C) NMS → Gradient → Gaussian smoothing → Hysteresis
- D) Hysteresis → NMS → Gradient → Gaussian smoothing

Answer: **C) NMS → Gradient → Gaussian smoothing → Hysteresis** ❌ (key B: Gaussian smoothing → Gradient → NMS → Hysteresis)
Evidence (nli): The Canny steps in order are Gaussian smoothing, gradient calculation, non-maximum suppression, then hysteresis thresholding. — `fact:0408 cv_fundamentals:canny`

**23. What is the purpose of Gaussian smoothing in Canny?**

- A) Reduces noise to prevent false edges
- B) Increases edge strength
- C) Thins edges
- D) Links weak edges

Answer: **A) Reduces noise to prevent false edges** ✅
Evidence (nli): Canny edge detection steps: Gaussian smoothing reduces noise, gradient calculation finds magnitude and direction, non-maximum suppression, and hysteresis thresholding. — `fact:0316 lab_manual_06:p11`

**24. What does Non-Maximum Suppression do in Canny?**

- A) Reduces noise
- B) Thins edges to 1-pixel width
- C) Links strong and weak edges
- D) Computes gradient magnitude

Answer: **B) Thins edges to 1-pixel width** ✅
Evidence (triple): Non-Maximum Suppression | does in Canny | thins edges to 1-pixel width — `triple:0114 lab_manual_06:p12`

**25. In hysteresis thresholding, weak edges are kept only if:**

- A) They are above the high threshold
- B) They are connected to strong edges
- C) They are isolated
- D) They are below the low threshold

Answer: **B) They are connected to strong edges** ✅
Evidence (nli): The manual says non-maximum suppression eliminates weak edge pixels and hysteresis tracks edges by linking strong edge pixels. — `fact:0317 lab_manual_06:p11`

**26. Canny edge detection is known for:**

- A) Thick and broken edges
- B) Thin and continuous edges
- C) Only horizontal edges
- D) Only vertical edges

Answer: **B) Thin and continuous edges** ✅
Evidence (nli): The Canny detector is known for high accuracy and for detecting thin and continuous edges. — `fact:0318 lab_manual_06:p11`

**27. LoG stands for:**

- A) Laplacian of Gaussian
- B) Line of Gradient
- C) Logarithm of Gradient
- D) Length of Gaussian

Answer: **A) Laplacian of Gaussian** ✅
Evidence (nli): Edge-based segmentation identifies rapid intensity changes at object boundaries using Sobel, Canny or Laplacian of Gaussian (LoG). — `fact:0249 lab_05_manual:p5`

**28. In LoG, edges are indicated by:**

- A) Peaks
- B) Zero-crossings
- C) Valleys
- D) High gradients

Answer: **B) Zero-crossings** ✅
Evidence (nli): The LoG operator first applies Gaussian smoothing and then computes the Laplacian; zero-crossings of the result indicate edges. — `fact:0322 lab_manual_06:p14`

**29. Which derivative does Canny/Sobel represent?**

- A) First derivative
- B) Second derivative
- C) Third derivative
- D) Zeroth derivative

Answer: **A) First derivative** ✅
Evidence (triple): Sobel | derivative | first derivative — `triple:0038 cv_fundamentals:derivatives`

**30. Which derivative does Laplacian represent?**

- A) First derivative
- B) Second derivative
- C) Third derivative
- D) Zeroth derivative

Answer: **B) Second derivative** ✅
Evidence (triple): Laplacian | derivative | second derivative — `triple:0039 cv_fundamentals:derivatives`

**31. Hough Transform is used to detect:**

- A) Lines and circles
- B) Only lines
- C) Only circles
- D) Only ellipses

Answer: **A) Lines and circles** ✅
Evidence (nli): The Hough transform detects simple shapes within an image, most commonly lines and circles. — `fact:0326 lab_manual_06:p17`

**32. Hough Transform transforms an image from spatial domain to:**

- A) Frequency domain
- B) Parameter space
- C) Color space
- D) Scale space

Answer: **B) Parameter space** ✅
Evidence (nli): The Hough transform maps the image spatial domain into a parameter space where each pixel corresponds to a curve or point. — `fact:0328 lab_manual_06:p17`

**33. A line in Cartesian coordinates is represented as:**

- A) y = mx + b
- B) y = ax² + bx + c
- C) x² + y² = r²
- D) y = mx² + b

Answer: **A) y = mx + b** ✅
Evidence (triple): line | represented as in Hough | y = mx + b — `triple:0115 lab_manual_06:p17`

**34. In Hough line detection, the parameter space axes are:**

- A) x and y
- B) m and b
- C) a, b, and r
- D) ρ and θ only

Answer: **B) m and b** ✅
Evidence (triple): Hough line | parameter space axes | m and b — `triple:0089 lab_manual_06:p17`

**35. In Hough Transform, each edge pixel:**

- A) Votes for possible shapes
- B) Is discarded
- C) Becomes a keypoint
- D) Is smoothed

Answer: **A) Votes for possible shapes** ✅
Evidence (nli): Hough circle voting picks potential centers and radii from edge pixels; peaks give circle centers and radii. — `fact:0334 lab_manual_06:p18`

**36. Peaks in the Hough parameter space represent:**

- A) Lines or circles
- B) Noise
- C) Gradients
- D) Keypoints

Answer: **A) Lines or circles** ✅
Evidence (nli): Hough circle voting picks potential centers and radii from edge pixels; peaks give circle centers and radii. — `fact:0334 lab_manual_06:p18`

**37. A circle in Cartesian coordinates is represented as:**

- A) y = mx + b
- B) (x − a)² + (y − b)² = r²
- C) x² + y² = r
- D) y = ax² + bx + c

Answer: **B) (x − a)² + (y − b)² = r²** ✅
Evidence (triple): circle | represented as in Hough | (x - a)^2 + (y - b)^2 = r^2 — `triple:0116 lab_manual_06:p18`

**38. The parameter space for circle detection has how many dimensions?**

- A) 1
- B) 2
- C) 3
- D) 4

Answer: **C) 3** ✅
Evidence (triple): circle detection | parameter space dimensions | 3 — `triple:0091 lab_manual_06:p18`

**39. In circle Hough Transform, the parameters are:**

- A) m and b
- B) a, b, and r
- C) x and y
- D) ρ and θ

Answer: **B) a, b, and r** ✅
Evidence (triple): circle Hough | parameters | a, b, and r — `triple:0090 lab_manual_06:p18`

**40. Who introduced the Hough Transform?**

- A) Paul Hough
- B) John Canny
- C) David Lowe
- D) Irvin Sobel

Answer: **A) Paul Hough** ✅
Evidence (triple): Hough Transform | introduced by | Paul Hough — `triple:0093 lab_manual_06:p17`

**41. The Hough Transform was first introduced in:**

- A) 1952
- B) 1962
- C) 1972
- D) 1982

Answer: **B) 1962** ✅
Evidence (triple): Hough Transform | introduced in | 1962 — `triple:0092 lab_manual_06:p17`

**42. Back-transformation in Hough Transform reveals:**

- A) Position and orientation of shapes
- B) Only color
- C) Only texture
- D) Only noise

Answer: **A) Position and orientation of shapes** ✅
Evidence (nli): Hough peak detection: peaks in the parameter space (accumulator) represent lines; back-transformation maps them to image positions. — `fact:0331 lab_manual_06:p17`

**43. SIFT stands for:**

- A) Scale-Invariant Feature Transform
- B) Scale-Invariant Fourier Transform
- C) Simple Invariant Feature Transform
- D) Spatial Invariant Feature Transform

Answer: **A) Scale-Invariant Feature Transform** ✅
Evidence (nli): SIFT (Scale-Invariant Feature Transform) is an algorithm for detecting and describing local features in images. — `fact:0150 lab_01_manual:p5`

**44. SIFT is used for:**

- A) Object recognition
- B) Image stitching
- C) Tracking
- D) All of the above

Answer: **A) Object recognition** ❌ (key D: All of the above)
Evidence (nli): SIFT (Scale-Invariant Feature Transform) extracts and matches features; useful for object recognition, image stitching and tracking. — `fact:0335 lab_manual_06:p18`

**45. DoG stands for:**

- A) Difference of Gaussians
- B) Derivative of Gradients
- C) Difference of Gradients
- D) Derivative of Gaussians

Answer: **A) Difference of Gaussians** ✅
Evidence (nli): SIFT step 1, scale-space extrema detection, finds keypoint candidates at multiple scales using a Difference of Gaussians (DoG). — `fact:0336 lab_manual_06:p18`

**46. The DoG is calculated as:**

- A) L(x, y, kσ) − L(x, y, σ)
- B) L(x, y, σ) + L(x, y, kσ)
- C) L(x, y, σ) × L(x, y, kσ)
- D) L(x, y, σ) / L(x, y, kσ)

Answer: **A) L(x, y, kσ) − L(x, y, σ)** ✅
Evidence (nli): The Difference of Gaussians (DoG) subtracts two Gaussian-blurred versions of the image: DoG = L(x, y, k sigma) - L(x, y, sigma). — `fact:0337 lab_manual_06:p18`

**47. Scale-space extrema detection finds:**

- A) Local extrema in scale space
- B) Global maxima only
- C) Zero-crossings
- D) Edges only

Answer: **A) Local extrema in scale space** ✅
Evidence (triple): scale-space extrema detection | finds | local extrema in scale space — `triple:0117 lab_manual_06:p18`

**48. In keypoint localization, a candidate keypoint is compared with how many neighbors?**

- A) 8
- B) 16
- C) 26
- D) 32

Answer: **C) 26** ✅
Evidence (triple): SIFT keypoint | compared with neighbors | 26 — `triple:0084 lab_manual_06:p19`

**49. The 26 neighbors consist of:**

- A) 8 same layer, 9 above, 9 below
- B) 9 same layer, 8 above, 9 below
- C) 8 same layer, 8 above, 10 below
- D) 10 same layer, 8 above, 8 below

Answer: **A) 8 same layer, 9 above, 9 below** ✅
Evidence (nli): A SIFT keypoint must be a local maximum or minimum compared to its 26 neighbors: 8 in the same scale, 9 in the scale above and 9 below. — `fact:0339 lab_manual_06:p19`

**50. Orientation assignment in SIFT makes keypoints invariant to:**

- A) Rotation
- B) Scale
- C) Illumination
- D) All of the above

Answer: **A) Rotation** ✅
Evidence (triple): orientation assignment | makes keypoints invariant to | rotation — `triple:0087 lab_manual_06:p20`

**51. The descriptor in SIFT is generated from:**

- A) 4×4 sub-regions and 8 orientation bins
- B) 3×3 sub-regions and 8 orientation bins
- C) 4×4 sub-regions and 6 orientation bins
- D) 8×8 sub-regions and 8 orientation bins

Answer: **A) 4×4 sub-regions and 8 orientation bins** ✅
Evidence (nli): The SIFT descriptor uses a 4x4 grid of sub-regions with 8 orientation bins each, giving 4 x 4 x 8 = 128 values. — `fact:0410 cv_fundamentals:sift`

**52. The SIFT descriptor size is:**

- A) 64
- B) 128
- C) 256
- D) 512

Answer: **B) 128** ✅
Evidence (triple): SIFT descriptor | descriptor size | 128 — `triple:0215 opencv_introspection:opencv-4.13.0`

**53. SIFT keypoint matching commonly uses:**

- A) Euclidean distance
- B) Manhattan distance
- C) Cosine similarity
- D) Hamming distance

Answer: **A) Euclidean distance** ✅
Evidence (nli): K-means assignment step: assign each pixel to the nearest centroid by Euclidean distance; update step: recompute each centroid as the mean of its pixels. — `fact:0286 lab_05_manual:p14`

**54. Outlier rejection in SIFT often uses:**

- A) RANSAC
- B) Sobel
- C) Canny
- D) Hough

Answer: **A) RANSAC** ✅
Evidence (triple): outlier rejection | uses in SIFT matching | RANSAC — `triple:0118 lab_manual_06:p21`

**55. Homography estimation is used for:**

- A) Image stitching
- B) Object recognition
- C) Both A and B
- D) Neither A nor B

Answer: **C) Both A and B** ✅
Evidence (nli): The Hough circle parameter space has three dimensions: a, b and r. — `fact:0333 lab_manual_06:p18`

**56. SIFT keypoints are invariant to:**

- A) Scale and rotation
- B) Only scale
- C) Only rotation
- D) Only translation

Answer: **A) Scale and rotation** ✅
Evidence (nli): SIFT keypoints are invariant to scale and rotation and partially robust to illumination and viewpoint changes. — `fact:0409 cv_fundamentals:sift`

**57. In SIFT, the descriptor is robust to:**

- A) Light changes
- B) Only rotation
- C) Only scale
- D) Only noise

Answer: **A) Light changes** ✅
Evidence (nli): The SIFT descriptor is a 128-dimensional vector (4x4 sub-regions x 8 orientation bins), robust to lighting changes. — `fact:0342 lab_manual_06:p21`

**58. The SIFT descriptor is formed by concatenating:**

- A) Histograms of gradient orientations
- B) Histograms of colors
- C) Histograms of intensities
- D) Histograms of edges

Answer: **A) Histograms of gradient orientations** ✅
Evidence (nli): SIFT descriptor generation builds histograms of gradient orientations in sub-regions around the keypoint and concatenates them. — `fact:0341 lab_manual_06:p20`

**59. Keypoint localization refines location by fitting:**

- A) 3D quadratic function
- B) 2D linear function
- C) 3D cubic function
- D) 2D quadratic function

Answer: **A) 3D quadratic function** ✅
Evidence (triple): keypoint localization | refines location by fitting | 3D quadratic function — `triple:0088 lab_manual_06:p19`

**60. SIFT detects keypoint candidates at:**

- A) Multiple scales
- B) Single scale
- C) Only the finest scale
- D) Only the coarsest scale

Answer: **A) Multiple scales** ✅
Evidence (nli): SIFT step 1, scale-space extrema detection, finds keypoint candidates at multiple scales using a Difference of Gaussians (DoG). — `fact:0336 lab_manual_06:p18`

**61. Which cv2.imread flag loads an image in grayscale?**

- A) cv2.IMREAD_COLOR
- B) cv2.IMREAD_GRAYSCALE
- C) cv2.IMREAD_UNCHANGED
- D) cv2.IMREAD_ANYCOLOR

Answer: **B) cv2.IMREAD_GRAYSCALE** ✅
Evidence (triple): imread | flag for grayscale | IMREAD_GRAYSCALE — `triple:0126 lab_01_manual:p24`

**62. What is the integer value of cv2.IMREAD_GRAYSCALE?**

- A) 1
- B) 0
- C) -1
- D) 2

Answer: **B) 0** ✅
Evidence (triple): cv2.IMREAD_GRAYSCALE | integer value | 0 — `triple:0130 opencv_introspection:opencv-4.13.0`

**63. Which cv2.cvtColor code converts BGR to RGB?**

- A) cv2.COLOR_BGR2GRAY
- B) cv2.COLOR_BGR2RGB
- C) cv2.COLOR_RGB2BGR
- D) cv2.COLOR_BGR2XYZ

Answer: **B) cv2.COLOR_BGR2RGB** ✅
Evidence (nli): Before plotting with Matplotlib, convert OpenCV's default BGR (Blue, Green, Red) to RGB with cv2.cvtColor(image, cv2.COLOR_BGR2RGB). — `fact:0179 lab_01_manual:p14`

**64. What is the integer value of cv2.COLOR_BGR2GRAY?**

- A) 4
- B) 6
- C) 8
- D) 10

Answer: **B) 6** ✅
Evidence (triple): cv2.COLOR_BGR2GRAY | integer value | 6 — `triple:0135 opencv_introspection:opencv-4.13.0`

**65. In cv2.Sobel, which depth is used to preserve negative values?**

- A) cv2.CV_8U
- B) cv2.CV_16S
- C) cv2.CV_32F
- D) cv2.CV_64F

Answer: **D) cv2.CV_64F** ✅
Evidence (nli): A signed float output depth such as cv2.CV_64F keeps negative gradient values that an 8-bit cv2.CV_8U output would clip to 0. — `fact:0363 opencv_docs:Sobel`

**66. In cv2.Sobel, what does ksize=3 mean?**

- A) 3×3 kernel
- B) 3×1 kernel
- C) 1×3 kernel
- D) 3×3 image

Answer: **A) 3×3 kernel** ✅
Evidence (nli): cv2.Sobel ksize=3 means a 3x3 Sobel kernel. — `fact:0362 opencv_docs:Sobel`

**67. In cv2.GaussianBlur, the kernel size must be:**

- A) Positive odd numbers
- B) Positive even numbers
- C) Any positive number
- D) Any integer

Answer: **A) Positive odd numbers** ✅
Evidence (nli): cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=0): ksize width and height must be positive and odd. — `fact:0354 opencv_docs:GaussianBlur`

**68. In the lab, what Gaussian blur kernel size is used?**

- A) (3, 3)
- B) (5, 5)
- C) (7, 7)
- D) (9, 9)

Answer: **B) (5, 5)** ✅
Evidence (triple): Gaussian blur | kernel size used in the lab | (5, 5) — `triple:0119 lab_manual_06:p13`

**69. In the lab, what sigmaX value is used for Gaussian blur?**

- A) 1.0
- B) 1.4
- C) 2.0
- D) 2.5

Answer: **B) 1.4** ✅
Evidence (triple): Gaussian blur | sigmaX value used in the lab | 1.4 — `triple:0120 lab_manual_06:p13`

**70. In cv2.Canny, what are the lower and upper thresholds used in the lab?**

- A) 30, 100
- B) 50, 150
- C) 100, 200
- D) 150, 250

Answer: **B) 50, 150** ✅
Evidence (triple): cv2.Canny | lower and upper thresholds used in the lab | 50, 150 — `triple:0121 lab_manual_06:p13`

**71. In cv2.Canny, what does apertureSize control?**

- A) Sobel operator aperture size
- B) Gaussian kernel size
- C) Threshold value
- D) Image depth

Answer: **A) Sobel operator aperture size** ✅
Evidence (triple): apertureSize | controls in cv2.Canny | Sobel operator aperture size — `triple:0044 opencv_docs:Canny`

**72. In cv2.Canny, if L2gradient=True, which norm is used?**

- A) L1 norm
- B) L2 norm
- C) L∞ norm
- D) L0 norm

Answer: **B) L2 norm** ✅
Evidence (triple): L2gradient=True | norm used | L2 norm — `triple:0045 opencv_docs:Canny`

**73. In cv2.Laplacian, which depth is used in the lab?**

- A) cv2.CV_8U
- B) cv2.CV_16S
- C) cv2.CV_32F
- D) cv2.CV_64F

Answer: **D) cv2.CV_64F** ✅
Evidence (triple): cv2.Laplacian | depth used in the lab | cv2.CV_64F — `triple:0122 lab_manual_06:p15`

**74. What does cv2.convertScaleAbs do?**

- A) Converts to absolute values and scales to 8-bit
- B) Converts to grayscale
- C) Converts to binary
- D) Converts to float

Answer: **A) Converts to absolute values and scales to 8-bit** ✅
Evidence (triple): cv2.convertScaleAbs | does | converts to absolute values and scales to 8-bit — `triple:0123 opencv_docs:convertScaleAbs`

**75. In cv2.convertScaleAbs, what does alpha control?**

- A) Scale factor (contrast)
- B) Delta (brightness)
- C) Kernel size
- D) Threshold

Answer: **A) Scale factor (contrast)** ✅
Evidence (nli): In cv2.convertScaleAbs, alpha is the optional scale factor (contrast) and beta is the delta added to the scaled values (brightness). — `fact:0368 opencv_docs:convertScaleAbs`

**76. In cv2.HoughCircles, which method is implemented?**

- A) cv2.HOUGH_STANDARD
- B) cv2.HOUGH_PROBABILISTIC
- C) cv2.HOUGH_GRADIENT
- D) cv2.HOUGH_MULTI_SCALE

Answer: **C) cv2.HOUGH_GRADIENT** ✅
Evidence (triple): cv2.HoughCircles | method implemented | cv2.HOUGH_GRADIENT — `triple:0056 opencv_docs:HoughCircles`

**77. In cv2.HoughCircles, what does minDist represent?**

- A) Minimum distance between centers of detected circles
- B) Minimum radius
- C) Maximum radius
- D) Accumulator threshold

Answer: **A) Minimum distance between centers of detected circles** ✅
Evidence (triple): minDist | represents in cv2.HoughCircles | minimum distance between centers of detected circles — `triple:0053 opencv_docs:HoughCircles`

**78. In cv2.HoughCircles, what does param2 represent?**

- A) Accumulator threshold for circle centers
- B) Higher threshold for Canny
- C) Minimum radius
- D) Maximum radius

Answer: **A) Accumulator threshold for circle centers** ✅
Evidence (triple): param2 | represents in cv2.HoughCircles | accumulator threshold for circle centers — `triple:0055 opencv_docs:HoughCircles`

**79. In cv2.SIFT_create, what is the default nOctaveLayers?**

- A) 1
- B) 2
- C) 3
- D) 4

Answer: **C) 3** ✅
Evidence (triple): SIFT_create | default nOctaveLayers | 3 — `triple:0210 opencv_introspection:opencv-4.13.0`

**80. In cv2.SIFT_create, what is the default contrastThreshold?**

- A) 0.01
- B) 0.03
- C) 0.04
- D) 0.05

Answer: **C) 0.04** ✅
Evidence (triple): SIFT_create | default contrastThreshold | 0.04 — `triple:0211 opencv_introspection:opencv-4.13.0`

**81. In cv2.SIFT_create, what is the default edgeThreshold?**

- A) 5
- B) 10
- C) 15
- D) 20

Answer: **B) 10** ✅
Evidence (triple): SIFT_create | default edgeThreshold | 10 — `triple:0212 opencv_introspection:opencv-4.13.0`

**82. In cv2.SIFT_create, what is the default sigma?**

- A) 1.0
- B) 1.4
- C) 1.6
- D) 2.0

Answer: **C) 1.6** ✅
Evidence (triple): SIFT_create | default sigma | 1.6 — `triple:0213 opencv_introspection:opencv-4.13.0`

**83. In cv2.SIFT_create, what is the default descriptorType?**

- A) CV_8U
- B) CV_16S
- C) CV_32F
- D) CV_64F

Answer: **C) CV_32F** ✅
Evidence (triple): SIFT_create | default descriptorType | CV_32F — `triple:0214 opencv_introspection:opencv-4.13.0`

**84. Which flag in cv2.drawKeypoints draws keypoints with size and orientation?**

- A) cv2.DRAW_MATCHES_FLAGS_DEFAULT
- B) cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
- C) cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
- D) cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

Answer: **B) cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS** ✅
Evidence (triple): DRAW_RICH_KEYPOINTS | draws keypoints with size and orientation | cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS — `triple:0066 opencv_docs:drawKeypoints`

**85. What is the shape of SIFT descriptors for N keypoints?**

- A) (N, 64)
- B) (N, 128)
- C) (128, N)
- D) (64, N)

Answer: **B) (N, 128)** ✅
Evidence (nli): sift.detectAndCompute returns keypoints and descriptors; descriptors has shape (N, 128) of dtype float32 for N keypoints. — `fact:0507 opencv_introspection:opencv-4.13.0`

**86. Which function detects keypoints and computes descriptors in SIFT?**

- A) sift.detect()
- B) sift.compute()
- C) sift.detectAndCompute()
- D) sift.keypoints()

Answer: **B) sift.compute()** ❌ (key C: sift.detectAndCompute())
Evidence (nli): The SIFT code creates the detector with cv2.SIFT_create() and calls sift.detectAndCompute(gray, None), returning keypoints and descriptors. — `fact:0347 lab_manual_06:p23`

**87. In cv2.HoughLines, what does rho represent?**

- A) Distance resolution in pixels
- B) Angle resolution in radians
- C) Accumulator threshold
- D) Minimum line length

Answer: **A) Distance resolution in pixels** ✅
Evidence (triple): rho | represents in cv2.HoughLines | distance resolution in pixels — `triple:0050 opencv_docs:HoughLines`

**88. In cv2.HoughLines, what does theta represent?**

- A) Distance resolution
- B) Angle resolution in radians
- C) Threshold
- D) Maximum angle

Answer: **B) Angle resolution in radians** ✅
Evidence (triple): theta | represents in cv2.HoughLines | angle resolution in radians — `triple:0051 opencv_docs:HoughLines`

**89. In cv2.HoughLines, what does threshold represent?**

- A) Accumulator threshold
- B) Minimum line length
- C) Maximum line gap
- D) Angle resolution

Answer: **A) Accumulator threshold** ✅
Evidence (triple): threshold | represents in cv2.HoughLines | accumulator threshold — `triple:0052 opencv_docs:HoughLines`

**90. In cv2.Sobel, what does dx=1, dy=0 compute?**

- A) Horizontal gradient (Gx)
- B) Vertical gradient (Gy)
- C) Laplacian
- D) Magnitude

Answer: **A) Horizontal gradient (Gx)** ✅
Evidence (triple): dx=1, dy=0 | computes in cv2.Sobel | horizontal gradient (Gx) — `triple:0036 opencv_docs:Sobel`

**91. In cv2.Sobel, what does dx=0, dy=1 compute?**

- A) Horizontal gradient (Gx)
- B) Vertical gradient (Gy)
- C) Laplacian
- D) Magnitude

Answer: **B) Vertical gradient (Gy)** ✅
Evidence (triple): dx=0, dy=1 | computes in cv2.Sobel | vertical gradient (Gy) — `triple:0037 opencv_docs:Sobel`

**92. Which function is used to compute gradient magnitude in the lab?**

- A) np.sqrt(sobel_x**2 + sobel_y**2)
- B) np.add(sobel_x, sobel_y)
- C) np.multiply(sobel_x, sobel_y)
- D) np.abs(sobel_x + sobel_y)

Answer: **A) np.sqrt(sobel_x**2 + sobel_y**2)** ✅
Evidence (nli): The HED code computes gradient_x and gradient_y with cv2.Sobel(..., cv2.CV_64F, ...) and magnitude np.sqrt(gradient_x**2 + gradient_y**2). — `fact:0058 lab_04_manual:p11`

**93. What does cv2.cvtColor(image, cv2.COLOR_BGR2RGB) do?**

- A) Converts BGR to RGB for Matplotlib
- B) Converts RGB to BGR
- C) Converts to grayscale
- D) Converts to binary

Answer: **A) Converts BGR to RGB for Matplotlib** ✅
Evidence (triple): COLOR_BGR2RGB | does in cv2.cvtColor | converts BGR to RGB for Matplotlib — `triple:0124 lab_01_manual:p14`

**94. In cv2.GaussianBlur, what happens if sigmaY=0?**

- A) It is set equal to sigmaX
- B) It is set to 1
- C) It is set to 0
- D) It causes an error

Answer: **A) It is set equal to sigmaX** ✅
Evidence (triple): sigmaY=0 | in cv2.GaussianBlur when 0 | set equal to sigmaX — `triple:0046 opencv_docs:GaussianBlur`

**95. In cv2.Laplacian, what is the default ksize?**

- A) 1
- B) 3
- C) 5
- D) 7

Answer: **A) 1** ✅
Evidence (triple): cv2.Laplacian | default ksize | 1 — `triple:0047 opencv_docs:Laplacian`

**96. Which edge detector uses hysteresis thresholding?**

- A) Sobel
- B) Canny
- C) LoG
- D) Hough

Answer: **B) Canny** ✅
Evidence (nli): Canny edge detection steps: Gaussian smoothing reduces noise, gradient calculation finds magnitude and direction, non-maximum suppression, and hysteresis thresholding. — `fact:0316 lab_manual_06:p11`

**97. Which edge detector uses zero-crossings?**

- A) Sobel
- B) Canny
- C) LoG
- D) Hough

Answer: **C) LoG** ✅
Evidence (nli): LoG stands for Laplacian of Gaussian; it combines Gaussian smoothing and Laplacian edge detection and highlights zero-crossings of the second derivative. — `fact:0105 lab_04_manual:p20`

**98. Which transform is used for line and circle detection?**

- A) Fourier Transform
- B) Hough Transform
- C) Wavelet Transform
- D) SIFT

Answer: **B) Hough Transform** ✅
Evidence (nli): The Hough transform detects simple shapes within an image, most commonly lines and circles. — `fact:0326 lab_manual_06:p17`

**99. Which feature detector is scale-invariant?**

- A) Sobel
- B) Canny
- C) SIFT
- D) Hough

Answer: **C) SIFT** ✅
Evidence (nli): SIFT (Scale-Invariant Feature Transform) extracts and matches features; useful for object recognition, image stitching and tracking. — `fact:0335 lab_manual_06:p18`

**100. Which algorithm is used for outlier rejection in SIFT matching?**

- A) RANSAC
- B) Sobel
- C) Canny
- D) LoG

Answer: **A) RANSAC** ✅
Evidence (triple): outlier rejection | uses in SIFT matching | RANSAC — `triple:0118 lab_manual_06:p21`
