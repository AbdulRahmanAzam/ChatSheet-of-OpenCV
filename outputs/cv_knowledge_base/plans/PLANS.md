# Plans for the course lab tasks

Generated offline by `work/planner.py` from the knowledge base. Each plan has the steps, why each step is there, its parameters, code, warnings and manual references, plus a full script (`.py`, same code) that was checked by `work/plan_selftest.py`.

| Task | Method | Steps | Files |
|---|---|---|---|
| Load the Sukuna image, display it with a title and hide the axes. | Basic image operations | 3 | [plan](load-the-sukuna-image-display-it.md), [script](load-the-sukuna-image-display-it.py) |
| Draw five concentric circles on the image and a bounding box around them. | Basic image operations | 3 | [plan](draw-five-concentric-circles-on-the.md), [script](draw-five-concentric-circles-on-the.py) |
| Apply a 25x25 blur to the image and extract the exact center region of interest. | Basic image operations | 4 | [plan](apply-a-25x25-blur-to-the.md), [script](apply-a-25x25-blur-to-the.py) |
| Add a transparent blue banner across the bottom of the image. | Basic image operations | 3 | [plan](add-a-transparent-blue-banner-across.md), [script](add-a-transparent-blue-banner-across.py) |
| Threshold the image and then rotate it 45 degrees with scaling. | Basic image operations | 5 | [plan](threshold-the-image-and-then-rotate.md), [script](threshold-the-image-and-then-rotate.py) |
| Compute the mean, standard deviation, min and max of each RGB channel. | Basic image operations | 3 | [plan](compute-the-mean-standard-deviation-min.md), [script](compute-the-mean-standard-deviation-min.py) |
| Enhance an X-ray image so the bones are clearer and apply pseudocolor. | Contrast enhancement and pseudocolor | 6 | [plan](enhance-an-x-ray-image-so.md), [script](enhance-an-x-ray-image-so.py) |
| Fuse a registered CT slice and MRI slice into one image. | Image fusion of two registered images | 3 | [plan](fuse-a-registered-ct-slice-and.md), [script](fuse-a-registered-ct-slice-and.py) |
| Visualize an echocardiogram video with better contrast. | Contrast enhancement and pseudocolor | 6 | [plan](visualize-an-echocardiogram-video-with-better.md), [script](visualize-an-echocardiogram-video-with-better.py) |
| Enlarge a fingerprint image using a manual scaling matrix. | Geometric transformation | 3 | [plan](enlarge-a-fingerprint-image-using-a.md), [script](enlarge-a-fingerprint-image-using-a.py) |
| Undo the rotation of a satellite image using sin and cos. | Geometric transformation | 3 | [plan](undo-the-rotation-of-a-satellite.md), [script](undo-the-rotation-of-a-satellite.py) |
| Remove the horizontal shear from a barcode image. | Geometric transformation | 3 | [plan](remove-the-horizontal-shear-from-a.md), [script](remove-the-horizontal-shear-from-a.py) |
| Recover the translation between two map images and shift it back. | Geometric transformation | 3 | [plan](recover-the-translation-between-two-map.md), [script](recover-the-translation-between-two-map.py) |
| Solve the six unknowns of an affine transform from three point pairs. | Geometric transformation | 3 | [plan](solve-the-six-unknowns-of-an.md), [script](solve-the-six-unknowns-of-an.py) |
| Rectify a square seen in perspective into a top-down view. | Geometric transformation | 3 | [plan](rectify-a-square-seen-in-perspective.md), [script](rectify-a-square-seen-in-perspective.py) |
| Build a panorama from two football field images using point correspondences. | Panorama stitching with SIFT and homography | 3 | [plan](build-a-panorama-from-two-football.md), [script](build-a-panorama-from-two-football.py) |
| Insert a painting into a wall photo using a perspective transform. | Geometric transformation | 3 | [plan](insert-a-painting-into-a-wall.md), [script](insert-a-painting-into-a-wall.py) |
| Analyse material texture using HOG and LBP. | Texture and shape features (HOG and LBP) | 6 | [plan](analyse-material-texture-using-hog-and.md), [script](analyse-material-texture-using-hog-and.py) |
| Compare global and adaptive thresholding on a document image. | Thresholding: global vs Otsu vs adaptive | 7 | [plan](compare-global-and-adaptive-thresholding-on.md), [script](compare-global-and-adaptive-thresholding-on.py) |
| Sweep the block size and C of adaptive thresholding on a document. | Thresholding: global vs Otsu vs adaptive | 7 | [plan](sweep-the-block-size-and-c.md), [script](sweep-the-block-size-and-c.py) |
| Segment coins with Otsu thresholding and show how the histogram changes. | Thresholding: global vs Otsu vs adaptive | 8 | [plan](segment-coins-with-otsu-thresholding-and.md), [script](segment-coins-with-otsu-thresholding-and.py) |
| Isolate the yellow car using HSV. | Color segmentation in HSV | 7 | [plan](isolate-the-yellow-car-using-hsv.md), [script](isolate-the-yellow-car-using-hsv.py) |
| Compare Canny with different high thresholds. | Edge detection with Canny | 5 | [plan](compare-canny-with-different-high-thresholds.md), [script](compare-canny-with-different-high-thresholds.py) |
| Region growing from two seeds with three tolerances. | Region growing from seeds | 4 | [plan](region-growing-from-two-seeds-with.md), [script](region-growing-from-two-seeds-with.py) |
| Implement the complete marker-based watershed pipeline. | Marker-based watershed segmentation | 7 | [plan](implement-the-complete-marker-based-watershed.md), [script](implement-the-complete-marker-based-watershed.py) |
| Show the effect of the distance-transform threshold on watershed. | Marker-based watershed segmentation | 7 | [plan](show-the-effect-of-the-distance.md), [script](show-the-effect-of-the-distance.py) |
| Segment an image with k-means for K=2,4,6. | Color clustering with k-means | 3 | [plan](segment-an-image-with-k-means.md), [script](segment-an-image-with-k-means.py) |
| You are working on a computer vision project for monitoring computer lab usage and ensu... | Screen detection and status with Hough lines | 12 | [plan](you-are-working-on-a-computer.md), [script](you-are-working-on-a-computer.py) |
| You are responsible for managing computer assets in a busy computing lab. Implement a c... | Object recognition with SIFT, ratio test and RANSAC homography | 6 | [plan](you-are-responsible-for-managing-computer.md), [script](you-are-responsible-for-managing-computer.py) |
| You are tasked with developing a system to monitor sensor data from industrial machines... | Wavelet denoising and anomaly detection on a signal | 3 | [plan](you-are-tasked-with-developing-a.md), [script](you-are-tasked-with-developing-a.py) |
| You have a reference image of the object you want to recognize and a set of test images... | Object recognition with SIFT, ratio test and RANSAC homography | 6 | [plan](you-have-a-reference-image-of.md), [script](you-have-a-reference-image-of.py) |
| Create a panoramic image by stitching multiple overlapping images captured from a camer... | Panorama stitching with SIFT and homography | 3 | [plan](create-a-panoramic-image-by-stitching.md), [script](create-a-panoramic-image-by-stitching.py) |
| You are working on an autonomous vehicle project, and one of the critical tasks is to d... | Lane-line detection with Hough lines | 9 | [plan](you-are-working-on-an-autonomous.md), [script](you-are-working-on-an-autonomous.py) |
| Count the coins in an image using the Hough circle transform. | Circle detection and counting with Hough circles | 6 | [plan](count-the-coins-in-an-image.md), [script](count-the-coins-in-an-image.py) |
| Monitor a restricted zone with a camera and raise an alert when someone enters it. | Restricted-zone change monitoring | 6 | [plan](monitor-a-restricted-zone-with-a.md), [script](monitor-a-restricted-zone-with-a.py) |
