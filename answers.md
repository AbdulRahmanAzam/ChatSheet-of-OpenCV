# Answer sheet: mcqs.txt

No answer key supplied.

The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.

**1. An image contains a long horizontal line and a short diagonal line. You apply: lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80) Which statement is most accurate?**

- A) Only horizontal lines can be detected
- B) Only lines longer than exactly 80 pixels are detected
- C) A line segment must accumulate enough votes and satisfy the minimum length constraint to be returned
- D) All detected lines will be converted into rectangles

Answer: **B) Only lines longer than exactly 80 pixels are detected**
Evidence (nli): cv2.HoughLinesP is the probabilistic Hough transform returning segment endpoints; minLineLength rejects shorter segments and maxLineGap joins segments with smaller gaps. — `fact:0390 opencv_docs:HoughLinesP`
