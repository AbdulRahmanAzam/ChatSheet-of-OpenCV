# Demo: what each command gives

Generated now from the real commands. Run them yourself from the project folder with `.venv\Scripts\python.exe`.

## 1. Conceptual questions: `worksk.py "question"`

### Why do we blur before Canny edge detection?

**Answer**

**In short:** Images are blurred before Canny because derivatives amplify noise: each noisy pixel creates a strong local gradient that would become a false edge, so smoothing first leaves only real intensity changes. Canny edge detection steps: Gaussian smoothing reduces noise, gradient calculation finds magnitude and direction, non-maximum suppression, and hysteresis thresholding.

A Canny pipeline smooths noise, computes gradients, thins local maxima along the gradient direction, then retains strong edges and weak edges connected to them. Low and high thresholds govern hysteresis. The result is a thin binary edge map, not filled object regions.

*When to use:* Prepare line detection or inspect boundaries.

**Key points from the course**

- Canny uses two thresholds (hysteresis) so that weak edge pixels are kept only when connected to strong ones; this keeps edges continuous while rejecting isolated noise responses. *[CV fundamentals, not from the manual]*
- Canny is a multi-stage edge detector known for its accuracy and noise reduction capabilities. *[Lab 04 Manual p.20]*
- The Canny detector is known for high accuracy and for detecting thin and continuous edges. *[Lab Manual 06 p.11]*
- Choosing Canny thresholds trades off edge detection sensitivity against noise suppression. *[Lab 04 Manual p.23]*

**Common mistakes**

- Use explicit blur to control smoothing. Non-maximum suppression is not simply a weak-pixel threshold. Final Canny output alone does not label which surviving pixels began strong or weak.


---

### Explain the ratio test in SIFT matching.

**Answer**

**In short:** Lowe's ratio test keeps a match only if the distance to the best match is clearly smaller than to the second-best (ratio below about 0.75); ambiguous matches, where two candidates are almost equally similar, are rejected. SIFT discards low-contrast keypoints (contrast threshold) and keypoints lying on edges (edge threshold, using the ratio of principal curvatures), because both are unstable.

A brute-force matcher compares descriptors. For float SIFT descriptors use L2 distance. Retrieve the two nearest candidates and retain a best match only when it is sufficiently better than the second; geometry then rejects additional outliers.

*When to use:* Reference recognition, inventory baselines and panorama alignment.

**Key points from the course**

- SIFT keypoint matching compares descriptors using Euclidean distance or other similarity measures. *[Lab Manual 06 p.21]*
- SIFT descriptors are matched with the L2 (Euclidean) norm; binary descriptors such as ORB use the Hamming norm. *[CV fundamentals, not from the manual]*
- Panorama stitching steps: detect keypoints and descriptors in overlapping images, match them with a ratio test, estimate a homography with RANSAC, warp one image into the other's frame, then blend the overlap. *[CV fundamentals, not from the manual]*
- SIFT (Scale-Invariant Feature Transform) extracts and matches features; useful for object recognition, image stitching and tracking. *[Lab Manual 06 p.18]*

**Common mistakes**

- The Python constructor is cv.BFMatcher, not cv.BruteForceMatcher. Guard empty descriptors and short pairs. Binary ORB descriptors normally use Hamming distance, with Hamming2 for ORB WTA_K=3/4.


---

### What is the difference between affine and perspective transformation?

**Answer**

**In short:** An affine transformation is a linear transformation followed by a translation: p' = Ap + b, where A is the linear matrix and b the translation vector. An affine transformation can change angles and distances but preserves the ratio of distances between points on a straight line.

**Linear, rigid, similarity and affine transformations.** A linear 2D map uses a 2x2 matrix and fixes the origin. A rigid map adds rotation and translation while preserving distances. Similarity adds uniform scaling. An affine map additionally allows nonuniform scaling and shear and preserves parallel lines. Three noncollinear point pairs determine a 2D affine map.

*When to use:* Geometric correction when perspective effects are absent or negligible.

**Perspective transformations and homographies.** A 3x3 homography maps between views of a plane or images related by pure camera rotation. It preserves straight lines but can change angles, length ratios, and parallelism. Four nondegenerate point pairs determine it up to scale.

*When to use:* Rectify documents/paintings, map a planar reference into a scene, or stitch suitable views.

**Key points from the course**

- An affine transformation preserves collinearity (points on a line stay on a line) and parallelism (parallel lines stay parallel). *[Lab 03 Manual p.11]*
- Affine versus perspective: affine keeps parallel lines parallel (2x3 matrix, 3 point pairs); perspective/projective can make parallel lines converge (3x3 matrix, 4 point pairs), modelling a change of camera viewpoint. *[CV fundamentals, not from the manual]*
- In an affine transformation the bottom row of the 3x3 matrix is fixed to [0, 0, 1], so the denominator is always 1. *[Lab 03 Manual p.19]*
- An affine transform has 6 unknowns (2x3 matrix), so 3 non-collinear point pairs determine it; it preserves parallel lines and ratios along a line but not angles or lengths. *[CV fundamentals, not from the manual]*

**Common mistakes**

- Affine does not preserve every length or angle. Three collinear points are degenerate. Same-size output can clip transformed content.
- Use consistent corner order. Arbitrary 3D scenes with parallax cannot be aligned by one homography. Degenerate geometry and outliers require rejection.


---

## 2. Task questions (plan + full code): `worksk.py "task"`

### Circle detection and counting with Hough circles

> Count the coins in an image using the Hough circle transform.

## Approach

Goal: detect each circle, draw it and count them. Input: one image. Method: Circle detection and counting with Hough circles.

## Steps

1. **Load the image and check it.** imread returns None (no exception) for a wrong path, so check it before using the image.
2. **Convert to grayscale.** Edge, line, circle and threshold operations work on one intensity channel.
3. **Remove speckle noise with a median blur.** A median blur removes salt-and-pepper noise and keeps edges sharp; HoughCircles is very sensitive to noise, so the manual blurs before it.
4. **Find circles with the Hough gradient method.** HOUGH_GRADIENT runs Canny internally (param1 = its high threshold) and votes for centers along gradient directions; param2 is the center-vote threshold (lower = more circles, more false ones).
5. **Draw circles and the count.** Outline + center for each detection, and the total written on the image.
6. **Show every stage side by side and save the result.** Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray".

## Code

```python
"""Circle detection and counting with Hough circles.

Task: Count the coins in an image using the Hough circle transform.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
MEDIAN_KSIZE = 5           # odd median kernel size
MIN_DIST = 40              # minimum distance between centers, pixels
CANNY_HIGH = 120           # param1: internal Canny high threshold
ACC_THRESHOLD = 30         # param2: center votes needed
MIN_RADIUS = 10            # pixels
MAX_RADIUS = 80            # pixels


def run(img):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages['Grayscale'] = gray

    # Step 3: Remove speckle noise with a median blur
    smooth = cv.medianBlur(gray, MEDIAN_KSIZE)
    stages['Median blurred'] = smooth

    # Step 4: Find circles with the Hough gradient method
    found = cv.HoughCircles(smooth, cv.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
                            param1=CANNY_HIGH, param2=ACC_THRESHOLD, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    circles = np.empty((0, 3), int) if found is None else np.round(found[0]).astype(int)
    report['count'] = len(circles)

    # Step 5: Draw circles and the count
    vis = img.copy()
    for x, y, r in circles:
        cv.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), 3)
    cv.putText(vis, f'Count: {len(circles)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    stages['Result'] = vis
    return stages, report


def show(original, stages):
    """Original plus every stage in one matplotlib figure (BGR converted to RGB)."""
    items = [('Original', original)] + [(k, v) for k, v in stages.items() if isinstance(v, np.ndarray) and v.ndim in (2, 3)]
    cols = min(3, len(items)); rows = (len(items) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, image) in enumerate(items, 1):
        plt.subplot(rows, cols, i)
        if image.ndim == 3:
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(name); plt.axis('off')
    plt.tight_layout(); plt.show()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'input.jpg'
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    stages, report = run(img)
    for name, value in report.items():
        print(f'{name}: {value}' if not isinstance(value, np.ndarray) else f'{name}: array {value.shape}')
    show(img, stages)
    if 'Result' in stages:
        cv.imwrite('result.png', stages['Result'])

if __name__ == '__main__':
    main()
```

## Limitations

Parameter meanings differ for other circle methods. Perspective turns circles into ellipses.

---

## 3. MCQs: `work\quiz.py mcqs.txt --out=answers.md`

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
Evidence (nli): A homography is a 3x3 matrix with 8 degrees of freedom (defined up to scale), so at least 4 point correspondences, no three collinear, are needed to compute it. — `fact:0563 cv_fundamentals:perspective`

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

## 4. Custom code (new, experimental): `work\codegen.py "Write solve(img) that ..."`

A small offline code model (Qwen2.5-Coder 0.5B) writes code for the exact question, then the code is checked
(parses, only cv2/numpy, every cv2 function exists, runs on a test image) and repaired up to 3 times.
If it still fails, you get the tested planner script instead, clearly marked.

**Status: not good enough yet.** First test on 3 of the 20 frozen custom tasks: 0 correct with the 0.5B model.
Typical mistakes it made:

- *"Count coins and return the largest radius"*: copied the reference code but forgot to return the answer.
- *"Mean brightness of the four quadrants"*: indexed a third channel on a grayscale image (crash).
- *"Rotate 90 degrees clockwise without cropping"*: rotated the wrong way and cropped the image.

Full 20-task results for 0.5B are running; the 1.5B model (much stronger at code, but 1.12 GB) is next.
