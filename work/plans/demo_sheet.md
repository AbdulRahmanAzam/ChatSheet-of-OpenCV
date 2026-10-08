# Answer sheet: demo_mcq.txt

Graded 4/4 correct.

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. Why is the image blurred before applying Canny?**

- A) To increase contrast
- B) To reduce noise that would create false edges
- C) To convert it to grayscale
- D) To make edges thicker

Answer: **B) To reduce noise that would create false edges** ✅
Evidence (nli): Images are blurred before Canny because derivatives amplify noise: each noisy pixel creates a strong local gradient that would become a false edge, so smoothing first leaves only real intensity changes. — `fact:0533 cv_fundamentals:canny`

**2. How many point pairs are needed to compute a homography?**

- A) 2
- B) 3
- C) 4
- D) 6

Answer: **C) 4** ✅
Evidence (nli): A homography is a 3x3 matrix with 8 degrees of freedom (defined up to scale), so at least 4 point correspondences, no three collinear, are needed to compute it. — `fact:0564 cv_fundamentals:perspective`

**3. Which OpenCV function separates touching objects using markers?**

- A) cv2.findContours
- B) cv2.watershed
- C) cv2.kmeans
- D) cv2.Canny

Answer: **B) cv2.watershed** ✅
Evidence (nli): _, markers = cv2.connectedComponents(sure_fg) labels the sure foreground; the second return value is the markers (labels) image. — `fact:0521 lab_05_manual:p13`

**4. Which of the following is NOT a step of the Canny edge detector?**

- A) Gaussian smoothing
- B) Non-maximum suppression
- C) Hysteresis thresholding
- D) K-means clustering

Answer: **D) K-means clustering** ✅
Evidence (nli): The manual says non-maximum suppression eliminates weak edge pixels and hysteresis tracks edges by linking strong edge pixels. — `fact:0317 lab_manual_06:p11`
