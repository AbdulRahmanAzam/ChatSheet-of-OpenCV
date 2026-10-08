# Answer sheet: mcq_lab01_lab03.txt

Graded 97/100 correct. Wrong: Q4, Q62, Q98

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. Computer vision primarily aims to:**

- A) Store images in databases
- B) Enable computers to interpret and understand visual information
- C) Compress images for the internet
- D) Convert analog signals to digital only

Answer: **B) Enable computers to interpret and understand visual information** ✅
Evidence (nli): Computer vision enables computers to interpret, analyze and understand visual information from the world, much like humans do. — `fact:0124 lab_01_manual:p4`

**2. Which of the following is NOT a basic terminology in computer vision?**

- A) Pixel
- B) Resolution
- C) Grayscale
- D) Compiler

Answer: **D) Compiler** ✅
Evidence (nli): Choosing appropriate feature extraction techniques and parameters is critical for successful computer vision applications. — `fact:0017 lab_04_manual:p6`

**3. A pixel is best defined as:**

- A) The largest unit of an image
- B) The smallest unit of an image carrying color/intensity
- C) A type of image filter
- D) A compression algorithm

Answer: **B) The smallest unit of an image carrying color/intensity** ✅
Evidence (nli): Pixel is short for picture element, the smallest unit of an image; each pixel carries color and intensity information. — `fact:0129 lab_01_manual:p4`

**4. Resolution of an image is typically expressed as:**

- A) File size in MB
- B) Dimensions in pixels, e.g., 1920×1080
- C) Number of colors
- D) Compression ratio

Answer: **D) Compression ratio** ❌ (key B: Dimensions in pixels, e.g., 1920×1080)
Evidence (nli): In cv2.HoughCircles, dp is the inverse ratio of accumulator resolution to image resolution and minDist is the minimum distance between the centers of detected circles. — `fact:0392 opencv_docs:HoughCircles`

**5. In a grayscale image, pixel values typically range from:**

- A) 0 to 100
- B) 0 to 255
- C) 1 to 10
- D) -255 to 255

Answer: **B) 0 to 255** ✅
Evidence (nli): A grayscale image has pixels representing only intensity, typically from 0 (black) to 255 (white). — `fact:0131 lab_01_manual:p4`

**6. RGB stands for:**

- A) Red, Gray, Blue
- B) Red, Green, Blue
- C) Red, Green, Black
- D) Ratio, Green, Blue

Answer: **B) Red, Green, Blue** ✅
Evidence (nli): RGB stands for Red, Green, Blue; each pixel is composed of values for these three primary colors. — `fact:0132 lab_01_manual:p4`

**7. Feature extraction is:**

- A) Removing all features from an image
- B) Identifying and isolating relevant features from an image
- C) Compressing an image
- D) Rotating an image

Answer: **B) Identifying and isolating relevant features from an image** ✅
Evidence (nli): Feature extraction is the process of identifying and isolating relevant features from an image, often using filters or algorithms. — `fact:0134 lab_01_manual:p5`

**8. Edge detection is used to:**

- A) Fill colors in an image
- B) Identify boundaries or transitions between regions
- C) Increase image resolution
- D) Convert RGB to grayscale

Answer: **B) Identify boundaries or transitions between regions** ✅
Evidence (nli): Edge detection identifies boundaries or transitions between different regions in an image. — `fact:0135 lab_01_manual:p5`

**9. Object detection means:**

- A) Identifying and localizing specific objects in an image/video
- B) Only classifying an image as cat or dog
- C) Removing background
- D) Rotating objects

Answer: **A) Identifying and localizing specific objects in an image/video** ✅
Evidence (nli): Object detection identifies and localizes specific objects within an image or video frame. — `fact:0136 lab_01_manual:p5`

**10. Segmentation is:**

- A) Dividing an image into meaningful segments or regions
- B) Increasing brightness
- C) Adding text to an image
- D) Blurring an image

Answer: **A) Dividing an image into meaningful segments or regions** ✅
Evidence (nli): Segmentation divides an image into meaningful segments or regions, often separating objects from the background. — `fact:0137 lab_01_manual:p5`

**11. A Convolutional Neural Network (CNN) is:**

- A) A type of database
- B) A deep learning model designed for grid-structured data like images
- C) A compression algorithm
- D) An image file format

Answer: **B) A deep learning model designed for grid-structured data like images** ✅
Evidence (nli): A Convolutional Neural Network (CNN) is a deep learning model for grid-structured data like images; convolutional layers automatically learn hierarchical features. — `fact:0139 lab_01_manual:p5`

**12. Deep learning is:**

- A) A subset of machine learning using neural networks with multiple layers
- B) A type of image resizing
- C) A thresholding technique
- D) A color conversion method

Answer: **A) A subset of machine learning using neural networks with multiple layers** ✅
Evidence (nli): Deep learning is a subset of machine learning using neural networks with multiple layers to solve complex tasks. — `fact:0140 lab_01_manual:p5`

**13. A feature map is:**

- A) A map of image coordinates
- B) Output generated by applying convolutional filters to an image
- C) A type of histogram
- D) A grayscale image

Answer: **B) Output generated by applying convolutional filters to an image** ✅
Evidence (nli): A feature map is the output generated by applying convolutional filters to an image. — `fact:0141 lab_01_manual:p5`

**14. Classification assigns:**

- A) A label or category to an input image
- B) A new coordinate system
- C) A blur kernel
- D) A compression ratio

Answer: **A) A label or category to an input image** ✅
Evidence (nli): Classification assigns a label or category to an input image, such as cat or dog. — `fact:0143 lab_01_manual:p5`

**15. Image registration is:**

- A) Aligning multiple images to a common coordinate system
- B) Registering an image in a database
- C) Adding text to an image
- D) Converting to grayscale

Answer: **A) Aligning multiple images to a common coordinate system** ✅
Evidence (nli): Image registration aligns multiple images of the same scene or object to a common coordinate system. — `fact:0144 lab_01_manual:p5`

**16. Optical flow estimates:**

- A) Motion of objects between consecutive video frames
- B) Color distribution
- C) Image resolution
- D) Compression rate

Answer: **A) Motion of objects between consecutive video frames** ✅
Evidence (nli): Optical flow estimates the motion of objects between consecutive frames in a video. — `fact:0145 lab_01_manual:p5`

**17. Tracking means:**

- A) Following movement of objects across multiple frames
- B) Detecting edges
- C) Enhancing contrast
- D) Cropping an image

Answer: **A) Following movement of objects across multiple frames** ✅
Evidence (nli): Tracking follows the movement of objects across multiple frames in a video. — `fact:0146 lab_01_manual:p5`

**18. Homography is:**

- A) A transformation relating two images of the same planar surface
- B) A type of blur
- C) A histogram equalization method
- D) A color space

Answer: **A) A transformation relating two images of the same planar surface** ✅
Evidence (nli): A homography is a transformation that relates two images of the same planar surface. — `fact:0147 lab_01_manual:p5`

**19. A feature descriptor is:**

- A) A compact representation of a feature for matching/recognition
- B) A large raw pixel array
- C) A type of image file
- D) A blur kernel

Answer: **A) A compact representation of a feature for matching/recognition** ✅
Evidence (nli): A feature descriptor is a compact representation of a feature used for matching or recognition. — `fact:0148 lab_01_manual:p5`

**20. A histogram represents:**

- A) Distribution of pixel intensities in an image
- B) Image resolution
- C) Number of channels
- D) Compression ratio

Answer: **A) Distribution of pixel intensities in an image** ✅
Evidence (nli): A histogram is a representation of the distribution of pixel intensities in an image. — `fact:0149 lab_01_manual:p5`

**21. SIFT stands for:**

- A) Scale-Invariant Feature Transform
- B) Simple Image Filtering Technique
- C) Scaled Intensity Frequency Transform
- D) Standard Image Format Type

Answer: **A) Scale-Invariant Feature Transform** ✅
Evidence (nli): SIFT (Scale-Invariant Feature Transform) is an algorithm for detecting and describing local features in images. — `fact:0150 lab_01_manual:p5`

**22. SURF is similar to SIFT but:**

- A) Slower
- B) Faster
- C) Only for grayscale
- D) Only for video

Answer: **B) Faster** ✅
Evidence (nli): SURF (Speeded-Up Robust Features) is a feature detection and description algorithm similar to SIFT but faster. — `fact:0151 lab_01_manual:p6`

**23. HOG stands for:**

- A) Histogram of Oriented Gradients
- B) High-Order Grayscale
- C) Horizontal Orientation Grid
- D) Histogram of RGB

Answer: **A) Histogram of Oriented Gradients** ✅
Evidence (nli): HOG stands for Histogram of Oriented Gradients. — `fact:0019 lab_04_manual:p6`

**24. Which is an application of computer vision?**

- A) Autonomous vehicles
- B) Medical imaging
- C) Facial recognition
- D) All of the above

Answer: **D) All of the above** ✅
Evidence (nli): In autonomous vehicles computer vision recognizes obstacles, pedestrians, traffic signs and lane markings for real-time driving decisions. — `fact:0154 lab_01_manual:p6`

**25. Which library is mainly used for general computer vision tasks?**

- A) OpenCV
- B) Keras
- C) Caffe
- D) Detectron2

Answer: **A) OpenCV** ✅
Evidence (nli): OpenCV is the library mainly used for general computer vision tasks; Keras is a high-level neural network API, Caffe a CNN deep learning framework and Detectron2 an object detection library. — `fact:0531 lab_01_manual:p7`

**26. TorchVision is primarily designed for:**

- A) PyTorch
- B) TensorFlow
- C) Caffe
- D) OpenVINO

Answer: **A) PyTorch** ✅
Evidence (nli): TorchVision is computer vision for PyTorch with native PyTorch integration. — `fact:0160 lab_01_manual:p7`

**27. Detectron2 is specialized in:**

- A) Object detection
- B) NLP
- C) Image compression
- D) Audio processing

Answer: **A) Object detection** ✅
Evidence (nli): OpenVINO (Intel) focuses on CV inference optimization; Detectron2 (PyTorch) specializes in object detection; Caffe is a deep learning framework focused on CNNs. — `fact:0161 lab_01_manual:p8`

**28. Hugging Face is known for:**

- A) Pretrained NLP and CV models
- B) Only image blurring
- C) Only thresholding
- D) Only image rotation

Answer: **A) Pretrained NLP and CV models** ✅
Evidence (triple): Hugging Face | known for | pretrained NLP and CV models — `triple:0110 lab_01_manual:p8`

**29. A digital image is defined as a function:**

- A) f(x, y)
- B) g(a, b, c)
- C) h(t)
- D) p(q)

Answer: **A) f(x, y)** ✅
Evidence (nli): When x, y and the amplitude values of f are all finite, discrete quantities, the image is a digital image. — `fact:0163 lab_01_manual:p10`

**30. In image coordinates, the origin is usually at:**

- A) Bottom-left
- B) Top-left
- C) Center
- D) Top-right

Answer: **B) Top-left** ✅
Evidence (triple): origin | location in image coordinates | top-left — `triple:0011 lab_03_manual:p6`

**31. In image coordinates, the x-axis goes:**

- A) Right
- B) Left
- C) Up
- D) Down

Answer: **A) Right** ✅
Evidence (triple): x | direction in image coordinates, goes, increases | right — `triple:0009 lab_03_manual:p6`

**32. In image coordinates, the y-axis goes:**

- A) Right
- B) Left
- C) Up
- D) Down

Answer: **D) Down** ✅
Evidence (triple): y | direction in image coordinates, goes, increases | down — `triple:0010 lab_03_manual:p6`

**33. In digital image representation, M represents:**

- A) Total rows / height
- B) Total columns / width
- C) Number of channels
- D) Intensity

Answer: **A) Total rows / height** ✅
Evidence (triple): M | represents in digital image | total rows / height — `triple:0007 lab_01_manual:p10`

**34. In digital image representation, N represents:**

- A) Total rows / height
- B) Total columns / width
- C) Number of channels
- D) Intensity

Answer: **B) Total columns / width** ✅
Evidence (triple): N | represents in digital image | total columns / width — `triple:0008 lab_01_manual:p10`

**35. The first fundamental step of Digital Image Processing is:**

- A) Image Enhancement
- B) Image Acquisition
- C) Segmentation
- D) Compression

Answer: **B) Image Acquisition** ✅
Evidence (nli): Image acquisition is the first step of the fundamental steps of digital image processing; pre-processing such as scaling is done here. — `fact:0167 lab_01_manual:p11`

**36. Image enhancement is used to:**

- A) Highlight interesting features like brightness and contrast
- B) Compress the image
- C) Segment the image
- D) Recognize objects

Answer: **A) Highlight interesting features like brightness and contrast** ✅
Evidence (nli): Image enhancement is the simplest and most attractive area of DIP; it highlights interesting features such as brightness and contrast. — `fact:0168 lab_01_manual:p12`

**37. Image restoration is used to:**

- A) Improve the appearance of an image
- B) Reduce image resolution
- C) Convert to binary
- D) Add noise

Answer: **A) Improve the appearance of an image** ✅
Evidence (nli): Image restoration is the stage in which the appearance of an image is improved. — `fact:0169 lab_01_manual:p12`

**38. Wavelets and multi-resolution processing are used for:**

- A) Data compression and pyramidal representation
- B) Color conversion
- C) Object detection
- D) Text addition

Answer: **A) Data compression and pyramidal representation** ✅
Evidence (nli): Wavelets and multi-resolution processing represent an image in various degrees of resolution, for data compression and pyramidal representation. — `fact:0171 lab_01_manual:p12`

**39. Compression is important because:**

- A) It reduces storage requirement
- B) It increases resolution
- C) It adds color
- D) It detects edges

Answer: **A) It reduces storage requirement** ✅
Evidence (nli): Compression reduces the storage requirement of an image; it is important for internet use. — `fact:0172 lab_01_manual:p12`

**40. Morphological processing deals with:**

- A) Extracting image components useful for shape representation
- B) Histogram equalization
- C) Color space conversion
- D) Image rotation

Answer: **A) Extracting image components useful for shape representation** ✅
Evidence (nli): Morphological processing provides tools for extracting image components useful in the representation and description of shape. — `fact:0173 lab_01_manual:p12`

**41. Which DIP step is considered the most difficult?**

- A) Image Acquisition
- B) Segmentation
- C) Compression
- D) Image Enhancement

Answer: **B) Segmentation** ✅
Evidence (nli): Segmentation partitions an image into its objects; it is the most difficult task in digital image processing. — `fact:0174 lab_01_manual:p12`

**42. Representation and description follow:**

- A) Segmentation
- B) Compression
- C) Acquisition
- D) Enhancement

Answer: **A) Segmentation** ✅
Evidence (nli): Representation and description follow segmentation; description extracts information to differentiate one class of objects from another. — `fact:0175 lab_01_manual:p13`

**43. Object recognition assigns:**

- A) Labels to objects based on descriptors
- B) New pixel values
- C) Compression ratios
- D) Blur kernels

Answer: **A) Labels to objects based on descriptors** ✅
Evidence (nli): Object recognition assigns a label to an object based on its descriptors. — `fact:0176 lab_01_manual:p13`

**44. The Knowledge Base in DIP:**

- A) Is the last stage and limits searching processes
- B) Is the first stage
- C) Only handles compression
- D) Replaces segmentation

Answer: **A) Is the last stage and limits searching processes** ✅
Evidence (nli): Knowledge base is the last stage of digital image processing; it locates important information and limits the search. — `fact:0177 lab_01_manual:p13`

**45. OpenCV loads color images in which format by default?**

- A) RGB
- B) BGR
- C) Grayscale
- D) HSV

Answer: **B) BGR** ✅
Evidence (triple): OpenCV | loads color images in format by default | BGR — `triple:0030 lab_01_manual:p14`

**46. Matplotlib expects images in which format?**

- A) BGR
- B) RGB
- C) Grayscale only
- D) Binary

Answer: **B) RGB** ✅
Evidence (triple): Matplotlib | expects image format | RGB — `triple:0029 lab_01_manual:p14`

**47. Which function converts BGR to RGB in OpenCV?**

- A) cv2.cvtColor()
- B) cv2.imread()
- C) cv2.resize()
- D) cv2.threshold()

Answer: **A) cv2.cvtColor()** ✅
Evidence (nli): cv2.cvtColor(src, code) converts colour spaces; cv2.COLOR_BGR2RGB converts OpenCV BGR to RGB for Matplotlib, COLOR_BGR2HSV converts to HSV and COLOR_BGR2GRAY to grayscale. — `fact:0351 opencv_docs:cvtColor`

**48. Grayscale conversion in OpenCV uses:**

- A) cv2.COLOR_BGR2GRAY
- B) cv2.COLOR_RGB2BGR
- C) cv2.COLOR_GRAY2BGR
- D) cv2.COLOR_BGR2HSV

Answer: **A) cv2.COLOR_BGR2GRAY** ✅
Evidence (triple): grayscale conversion | code used in OpenCV | cv2.COLOR_BGR2GRAY — `triple:0125 lab_01_manual:p15`

**49. In cv2.resize(), fx and fy represent:**

- A) Scale factors along x and y
- B) Filter types
- C) Frame rates
- D) Font sizes

Answer: **A) Scale factors along x and y** ✅
Evidence (nli): cv2.resize(src, dsize, fx, fy, interpolation): dsize is (width, height); fx and fy are scale factors along x and y used when dsize is None. — `fact:0402 opencv_docs:resize`

**50. In GaussianBlur, the kernel size must be:**

- A) Even numbers
- B) Odd numbers
- C) Any integer
- D) Only 3

Answer: **B) Odd numbers** ✅
Evidence (nli): cv2.GaussianBlur(image, (31, 31), 0) blurs with a 31x31 kernel; kernel sizes must be odd numbers. — `fact:0186 lab_01_manual:p17`

**51. Cropping an image in OpenCV is done using:**

- A) NumPy slicing
- B) cv2.crop()
- C) cv2.cut()
- D) cv2.slice()

Answer: **A) NumPy slicing** ✅
Evidence (nli): Cropping in OpenCV uses NumPy array slicing image[y_start:y_end, x_start:x_end], not an OpenCV function. — `fact:0188 lab_01_manual:p18`

**52. cv2.putText() places text using which reference point?**

- A) Bottom-left corner of the text
- B) Top-left corner of the image
- C) Center of the image
- D) Top-right corner of the text

Answer: **A) Bottom-left corner of the text** ✅
Evidence (nli): cv2.putText(image, text, (x, y), font, fontScale, color, thickness) draws text; (x, y) is the bottom-left corner of the text. — `fact:0190 lab_01_manual:p19`

**53. In Pandas image analysis, image.reshape(-1, 3) converts:**

- A) 3D image array to 2D array
- B) 2D array to 3D array
- C) Grayscale to RGB
- D) RGB to BGR

Answer: **A) 3D image array to 2D array** ✅
Evidence (triple): reshape(-1, 3) | converts | 3D image array to 2D array — `triple:0033 lab_01_manual:p20`

**54. cv2.THRESH_BINARY does:**

- A) If pixel > threshold, set to max value; else set to 0
- B) If pixel < threshold, set to max value; else set to 0
- C) Keeps all pixels unchanged
- D) Inverts all pixels

Answer: **A) If pixel > threshold, set to max value; else set to 0** ✅
Evidence (nli): With THRESH_BINARY, pixels strictly greater than the threshold become the max value and all others become 0 (black). — `fact:0195 lab_01_manual:p21`

**55. cv2.getRotationMatrix2D() requires:**

- A) Center, angle, scale
- B) Width, height, channels
- C) Threshold, max value, type
- D) Kernel size, sigma

Answer: **A) Center, angle, scale** ✅
Evidence (nli): cv2.getRotationMatrix2D(center, angle, scale) computes a 2x3 rotation matrix; positive angles rotate counter-clockwise. — `fact:0197 lab_01_manual:p22`

**56. cv2.warpAffine() uses which matrix size?**

- A) 2×2
- B) 2×3
- C) 3×3
- D) 4×4

Answer: **B) 2×3** ✅
Evidence (triple): warpAffine | matrix size, accepts, uses | 2x3 — `triple:0012 lab_03_manual:p8`

**57. cv2.add() uses:**

- A) Saturation arithmetic
- B) Modulo arithmetic
- C) Subtraction
- D) Division

Answer: **A) Saturation arithmetic** ✅
Evidence (nli): cv2.add performs saturation arithmetic: sums above 255 are capped at 255 instead of wrapping around, so the result looks brighter. — `fact:0200 lab_01_manual:p23`

**58. Histogram equalization improves:**

- A) Contrast
- B) Resolution
- C) File size
- D) Color depth

Answer: **A) Contrast** ✅
Evidence (nli): cv2.equalizeHist spreads clumped pixel intensities across the full 0-255 range, improving global contrast of dark or washed-out images. — `fact:0202 lab_01_manual:p24`

**59. cv2.IMREAD_GRAYSCALE loads an image:**

- A) Directly as grayscale
- B) As RGB
- C) As BGR
- D) As binary only

Answer: **A) Directly as grayscale** ✅
Evidence (nli): cv2.imread(path, cv2.IMREAD_GRAYSCALE) loads an image directly as single-channel grayscale. — `fact:0201 lab_01_manual:p24`

**60. A geometric transformation changes:**

- A) Pixel color
- B) Spatial position, orientation, scale, or shape
- C) Only brightness
- D) Only contrast

Answer: **B) Spatial position, orientation, scale, or shape** ✅
Evidence (nli): A geometric transformation is a mathematical operation that alters the spatial position, orientation, scale or shape of an object without changing its underlying data values. — `fact:0203 lab_03_manual:p3`

**61. A photometric transformation changes:**

- A) Where the pixel is located
- B) What the pixel looks like (color, brightness, contrast)
- C) Image dimensions
- D) Coordinate system

Answer: **B) What the pixel looks like (color, brightness, contrast)** ✅
Evidence (nli): Geometric transformations change where a pixel is located; photometric transformations change what the pixel looks like (color, brightness or contrast). — `fact:0206 lab_03_manual:p4`

**62. Geometric transformation formula is:**

- A) I(x,y) → I'(x',y')
- B) (x,y) → (x',y')
- C) f(x,y) → g(x,y)
- D) p = Ap + b

Answer: **A) I(x,y) → I'(x',y')** ❌ (key B: (x,y) → (x',y'))
Evidence (nli): A geometric transformation changes where image information is located, mapping I(x, y) to I(x', y'). — `fact:0204 lab_03_manual:p3`

**63. Photometric transformation formula is:**

- A) (x,y) → (x',y')
- B) I(x,y) → I'(x',y')
- C) A0 = 0
- D) p' = Ap

Answer: **B) I(x,y) → I'(x',y')** ✅
Evidence (nli): An image can be viewed as a function I(x, y) where x is the horizontal coordinate, y the vertical coordinate and I the pixel intensity or color. — `fact:0205 lab_03_manual:p3`

**64. In image coordinate system, the origin is:**

- A) Bottom-left
- B) Top-left
- C) Center
- D) Top-right

Answer: **B) Top-left** ✅
Evidence (nli): In the image coordinate convention the origin (0,0) is the top-left corner; M is the total rows (height) and N the total columns (width). — `fact:0164 lab_01_manual:p10`

**65. In image coordinate system, x increases:**

- A) Right
- B) Left
- C) Up
- D) Down

Answer: **A) Right** ✅
Evidence (nli): In image coordinates the origin (0,0) is the top-left corner, x increases to the right and y increases downward, unlike Cartesian coordinates. — `fact:0210 lab_03_manual:p6`

**66. In image coordinate system, y increases:**

- A) Right
- B) Left
- C) Up
- D) Down

Answer: **D) Down** ✅
Evidence (nli): In image coordinates the origin (0,0) is the top-left corner, x increases to the right and y increases downward, unlike Cartesian coordinates. — `fact:0210 lab_03_manual:p6`

**67. A 2D transformation operates on:**

- A) (x,y) → (x',y')
- B) (x,y,z) → (x',y',z')
- C) Only 3D points
- D) Only color values

Answer: **A) (x,y) → (x',y')** ✅
Evidence (nli): An image is a two-dimensional function f(x, y) where x and y are spatial coordinates and the amplitude f is the intensity or gray level at that point. — `fact:0162 lab_01_manual:p10`

**68. OpenCV's warpAffine() accepts a:**

- A) 2×3 matrix
- B) 3×3 matrix
- C) 4×4 matrix
- D) 1×2 matrix

Answer: **A) 2×3 matrix** ✅
Evidence (nli): OpenCV's warpAffine() accepts a 2x3 matrix while warpPerspective() uses a 3x3 matrix. — `fact:0213 lab_03_manual:p8`

**69. OpenCV's warpPerspective() uses a:**

- A) 2×2 matrix
- B) 2×3 matrix
- C) 3×3 matrix
- D) 4×4 matrix

Answer: **C) 3×3 matrix** ✅
Evidence (nli): OpenCV's warpAffine() accepts a 2x3 matrix while warpPerspective() uses a 3x3 matrix. — `fact:0213 lab_03_manual:p8`

**70. A 3D homogeneous transformation typically uses a:**

- A) 2×3 matrix
- B) 3×3 matrix
- C) 4×4 matrix
- D) 2×2 matrix

Answer: **C) 4×4 matrix** ✅
Evidence (nli): A 3D transformation works on (x, y, z) in 3D world space and uses a 4x4 homogeneous matrix, used in robotics and 3D graphics. — `fact:0214 lab_03_manual:p9`

**71. In a 2×2 transformation matrix, the first column represents:**

- A) New landing spot for the X-axis
- B) New landing spot for the Y-axis
- C) Translation in x
- D) Translation in y

Answer: **A) New landing spot for the X-axis** ✅
Evidence (triple): first column | represents in 2x2 transformation matrix | new landing spot for the X-axis — `triple:0005 lab_03_manual:p10`

**72. In a 2×2 transformation matrix, the second column represents:**

- A) New landing spot for the X-axis
- B) New landing spot for the Y-axis
- C) Translation in x
- D) Translation in y

Answer: **B) New landing spot for the Y-axis** ✅
Evidence (triple): second column | represents in 2x2 transformation matrix | new landing spot for the Y-axis — `triple:0006 lab_03_manual:p10`

**73. In the matrix [[a, b], [c, d]], variable 'a' controls:**

- A) Horizontal scaling
- B) Vertical shearing
- C) Horizontal shearing
- D) Vertical scaling

Answer: **A) Horizontal scaling** ✅
Evidence (triple): a | controls in matrix [[a, b], [c, d]] | horizontal scaling — `triple:0001 lab_03_manual:p10`

**74. In the matrix [[a, b], [c, d]], variable 'c' controls:**

- A) Horizontal scaling
- B) Vertical shearing
- C) Horizontal shearing
- D) Vertical scaling

Answer: **B) Vertical shearing** ✅
Evidence (triple): c | controls in matrix [[a, b], [c, d]] | vertical shearing — `triple:0003 lab_03_manual:p10`

**75. In the matrix [[a, b], [c, d]], variable 'b' controls:**

- A) Horizontal scaling
- B) Vertical shearing
- C) Horizontal shearing
- D) Vertical scaling

Answer: **C) Horizontal shearing** ✅
Evidence (triple): b | controls in matrix [[a, b], [c, d]] | horizontal shearing — `triple:0002 lab_03_manual:p10`

**76. In the matrix [[a, b], [c, d]], variable 'd' controls:**

- A) Horizontal scaling
- B) Vertical shearing
- C) Horizontal shearing
- D) Vertical scaling

Answer: **D) Vertical scaling** ✅
Evidence (triple): d | controls in matrix [[a, b], [c, d]] | vertical scaling — `triple:0004 lab_03_manual:p10`

**77. An affine transformation preserves:**

- A) Collinearity and parallelism
- B) Only color
- C) Only brightness
- D) Only resolution

Answer: **A) Collinearity and parallelism** ✅
Evidence (nli): An affine transformation preserves collinearity (points on a line stay on a line) and parallelism (parallel lines stay parallel). — `fact:0222 lab_03_manual:p11`

**78. The affine transformation equation is:**

- A) p' = Ap
- B) p' = Ap + b
- C) p' = A + p
- D) p' = A/p

Answer: **B) p' = Ap + b** ✅
Evidence (triple): affine transformation | equation | p' = Ap + b — `triple:0018 lab_03_manual:p11`

**79. A linear transformation is an affine transformation:**

- A) With translation
- B) Without translation
- C) With perspective
- D) With color change

Answer: **B) Without translation** ✅
Evidence (nli): A linear transformation is an affine transformation without a translation vector; it uses only matrix multiplication and is tethered to the origin. — `fact:0226 lab_03_manual:p13`

**80. A linear transformation is represented by:**

- A) p' = Ap
- B) p' = Ap + b
- C) p' = A + b
- D) p' = A - b

Answer: **A) p' = Ap** ✅
Evidence (triple): linear transformation | equation, represented by | p' = Ap — `triple:0017 lab_03_manual:p13`

**81. The limitation of linear transformation is that it:**

- A) Cannot rotate
- B) Cannot scale
- C) Cannot translate because origin is anchored
- D) Cannot shear

Answer: **C) Cannot translate because origin is anchored** ✅
Evidence (triple): linear transformation | limitation | cannot translate because origin is anchored — `triple:0020 lab_03_manual:p16`

**82. Homogeneous coordinates add which value to (x, y)?**

- A) 0
- B) 1
- C) -1
- D) 2

Answer: **B) 1** ✅
Evidence (nli): Homogeneous coordinates add a dummy 1, turning (x, y) into (x, y, 1), so translation can be done by matrix multiplication alone. — `fact:0233 lab_03_manual:p17`

**83. Homogeneous coordinates turn (x, y) into:**

- A) (x, y, 0)
- B) (x, y, 1)
- C) (x, y, -1)
- D) (x, y, 2)

Answer: **B) (x, y, 1)** ✅
Evidence (nli): Homogeneous coordinates add a dummy 1, turning (x, y) into (x, y, 1), so translation can be done by matrix multiplication alone. — `fact:0233 lab_03_manual:p17`

**84. Homogeneous coordinates allow:**

- A) Translation using matrix multiplication
- B) Only rotation
- C) Only scaling
- D) Only reflection

Answer: **A) Translation using matrix multiplication** ✅
Evidence (nli): Homogeneous coordinates add a dummy 1, turning (x, y) into (x, y, 1), so translation can be done by matrix multiplication alone. — `fact:0233 lab_03_manual:p17`

**85. A projective transformation is also called:**

- A) Homography
- B) Histogram
- C) Convolution
- D) Thresholding

Answer: **A) Homography** ✅
Evidence (triple): projective transformation | also called | homography — `triple:0024 lab_03_manual:p18`

**86. A projective transformation uses a:**

- A) 2×2 matrix
- B) 2×3 matrix
- C) 3×3 matrix
- D) 4×4 matrix

Answer: **C) 3×3 matrix** ✅
Evidence (nli): A projective transformation uses a full 3x3 matrix H; the result is normalized by dividing by the third coordinate w. — `fact:0239 lab_03_manual:p19`

**87. Projective transformations allow:**

- A) Parallel lines to remain parallel
- B) Parallel lines to converge
- C) Only rotation
- D) Only scaling

Answer: **B) Parallel lines to converge** ✅
Evidence (nli): Projective transformations (homographies) simulate how cameras perceive 3D depth in 2D; they allow parallel lines to converge, creating perspective. — `fact:0237 lab_03_manual:p18`

**88. In affine transformation, the bottom row of the matrix is:**

- A) [1, 1, 1]
- B) [0, 0, 1]
- C) [0, 1, 0]
- D) [1, 0, 0]

Answer: **B) [0, 0, 1]** ✅
Evidence (nli): In an affine transformation the bottom row of the 3x3 matrix is fixed to [0, 0, 1], so the denominator is always 1. — `fact:0241 lab_03_manual:p19`

**89. The denominator effect in projective transformation creates:**

- A) Depth/perspective
- B) Color change
- C) Blur
- D) Noise

Answer: **A) Depth/perspective** ✅
Evidence (nli): In a projective transformation the denominator varies with x and y; points further away get a larger denominator and shrink together, which generates depth. — `fact:0242 lab_03_manual:p19`

**90. Projective transformation can model:**

- A) Converging lines and camera viewpoints
- B) Only 2D rotation
- C) Only grayscale conversion
- D) Only thresholding

Answer: **A) Converging lines and camera viewpoints** ✅
Evidence (nli): Projective transformations model converging lines (a road narrowing at the horizon), camera viewpoint changes and planar alignment. — `fact:0243 lab_03_manual:p20`

**91. Planar alignment uses which transformation?**

- A) Linear
- B) Affine
- C) Projective
- D) Photometric

Answer: **C) Projective** ✅
Evidence (nli): Projective transformations model converging lines (a road narrowing at the horizon), camera viewpoint changes and planar alignment. — `fact:0243 lab_03_manual:p20`

**92. The key difference between affine and projective is:**

- A) Affine preserves parallelism; projective does not
- B) Affine changes color; projective does not
- C) Affine is 3D; projective is 2D
- D) Affine uses 4×4; projective uses 2×3

Answer: **A) Affine preserves parallelism; projective does not** ✅
Evidence (nli): Affine transformations always preserve parallelism; projective transformations do not. — `fact:0238 lab_03_manual:p18`

**93. Linear transformations can achieve:**

- A) Scaling, rotation, reflection
- B) Translation only
- C) Perspective only
- D) Histogram equalization

Answer: **A) Scaling, rotation, reflection** ✅
Evidence (triple): linear transformation | can achieve | scaling, rotation, reflection, shearing — `triple:0019 lab_03_manual:p14`

**94. In a homogeneous 3×3 matrix, translation values are placed in:**

- A) The rightmost column
- B) The bottom row
- C) The diagonal
- D) The first row

Answer: **A) The rightmost column** ✅
Evidence (triple): homogeneous 3x3 matrix | translation placed in | the rightmost column — `triple:0022 lab_03_manual:p17`

**95. Composition of transformations is powerful because:**

- A) Multiple transformations can be combined into one master matrix
- B) It removes all noise
- C) It converts RGB to BGR
- D) It increases resolution

Answer: **A) Multiple transformations can be combined into one master matrix** ✅
Evidence (nli): With homogeneous coordinates, scale, rotation and translation matrices can be multiplied into one master matrix and applied in a single multiplication (composition). — `fact:0236 lab_03_manual:p18`

**96. In OpenCV, positive rotation angle in getRotationMatrix2D is:**

- A) Clockwise
- B) Counter-clockwise
- C) Always 0
- D) Always 180

Answer: **B) Counter-clockwise** ✅
Evidence (nli): cv2.getRotationMatrix2D(center, angle, scale) computes a 2x3 rotation matrix; positive angles rotate counter-clockwise. — `fact:0197 lab_01_manual:p22`

**97. Which of the following is a geometric transformation?**

- A) Brightness adjustment
- B) Rotation
- C) Gamma correction
- D) Color conversion

Answer: **B) Rotation** ✅
Evidence (nli): A geometric transformation is a mathematical operation that alters the spatial position, orientation, scale or shape of an object without changing its underlying data values. — `fact:0203 lab_03_manual:p3`

**98. Which of the following is a photometric transformation?**

- A) Translation
- B) Scaling
- C) Contrast enhancement
- D) Shearing

Answer: **A) Translation** ❌ (key C: Contrast enhancement)
Evidence (nli): Watershed preprocessing converts the input image to grayscale and may apply noise reduction or contrast enhancement. — `fact:0523 lab_05_manual:p12`

**99. The image coordinate system is important because:**

- A) It affects how transformations like rotation and cropping are applied
- B) It changes pixel colors
- C) It compresses images
- D) It equalizes histograms

Answer: **A) It affects how transformations like rotation and cropping are applied** ✅
Evidence (triple): image coordinate system | important because | it affects how transformations like rotation and cropping are applied — `triple:0111 lab_03_manual:p6`

**100. A 2D affine transformation can be represented as:**

- A) 2×2 only
- B) 2×3 or 3×3 homogeneous
- C) 4×4 only
- D) 1×2 only

Answer: **B) 2×3 or 3×3 homogeneous** ✅
Evidence (triple): 2D affine transformation | represented as | 2x3 or 3x3 homogeneous — `triple:0016 lab_03_manual:p9`
