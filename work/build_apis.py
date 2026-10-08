from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib,inspect,json
ROOT=Path(__file__).resolve().parents[1]/'outputs'/'cv_knowledge_base'
data='''imread|io|Decode local file; flag selects color/grayscale|BGR or grayscale array; None on failure
imwrite|io|Filename extension selects encoder|Boolean success
cvtColor|colors|Source channel layout must match conversion code|Converted array
resize|resize|dsize=(width,height); fx/fy alternative|Resampled array
circle|drawing|center(x,y), integer radius, BGR color|Mutates image
rectangle|drawing|Two corners or rectangle; BGR color|Mutates image
line|drawing|Two (x,y) endpoints|Mutates image
putText|drawing|Text baseline origin, font, scale and thickness|Mutates image
add|blend|Matching sizes/types or supported scalar|Saturating addition for uint8
addWeighted|blend|Matching arrays; alpha,beta,gamma offset|Weighted sum
bitwise_and|mask|Optional uint8 mask matches spatial shape|Bitwise result, not arithmetic weighting
bitwise_or|mask|Matching arrays|Bitwise union of disjoint selected images
bitwise_not|mask|uint8 mask for binary inverse|Bit complement
inRange|mask|Lower and upper scalar/channel bounds|uint8 selection mask
calcHist|histogram|List of images, channels, mask, bin counts and ranges|Histogram array
equalizeHist|equalization|uint8 single channel|Equalized uint8 grayscale
createCLAHE|equalization|clipLimit and tileGridSize|CLAHE object with apply method
LUT|gamma|256-value lookup table for byte inputs|Point-transformed image
applyColorMap|pseudocolor|Use uint8 scalar image and colormap identifier|BGR pseudocolor
filter2D|kernel|Kernel, ddepth, anchor, borderType; no strides parameter|Same spatial size correlation output
sepFilter2D|gaussian|Horizontal and vertical 1D kernels|Separable correlation result
blur|box|Kernel width,height|Normalized box average
boxFilter|box|normalize determines average versus sum|Local box response
GaussianBlur|gaussian|Odd kernel width,height and sigma|Smoothed image
getGaussianKernel|gaussian|Odd kernel size and sigma|1D column kernel
medianBlur|median|Odd aperture greater than1|Median-filtered image
bilateralFilter|bilateral|Diameter, color sigma, spatial sigma|Edge-preserving smoothed image
copyMakeBorder|padding|top,bottom,left,right and border rule|Larger padded array
Sobel|gradients|ddepth,dx,dy,ksize; float depth preserves sign|Signed derivative if depth permits
Scharr|scharr|First derivative dx,dy|Compact gradient response
Laplacian|laplacian|Signed ddepth recommended|Second derivative sum
convertScaleAbs|convert_abs|src,alpha,beta|Absolute scaled uint8; sign lost
Canny|canny|Grayscale uint8,low,high and optional L2gradient|Binary thin edge image
threshold|threshold|Single-channel input,cutoff,maxval,type|Selected threshold and output image
adaptiveThreshold|adaptive|uint8 gray; blockSize odd>1; C subtraction|Binary adaptive mask
getStructuringElement|morphology|Shape identifier and kernel size|Morphology kernel
erode|morphology|Image/mask and structuring element|Eroded array
dilate|morphology|Image/mask and structuring element|Dilated array
morphologyEx|morphology|Operation identifier and element|Opening/closing/gradient/top-hat/black-hat
distanceTransform|distance|Binary uint8 foreground nonzero|Distance to nearest zero
connectedComponents|components|Binary single-channel image|Count including background and label array
connectedComponentsWithStats|components|Binary single-channel image|Count,labels,stats,centroids
watershed|watershed|uint8 BGR source,int32 marker array|Mutates markers; boundary label-1
kmeans|kmeans|N-by-D float32 samples,K,criteria,attempts,flags|Compactness,labels,centers
setRNGSeed|kmeans|Integer seed|Resets OpenCV random generator
findContours|contours|Binary mask,retrieval mode,approximation mode|Contours and hierarchy
contourArea|contours|Point contour|Geometric area
arcLength|contours|Curve points,closed flag|Perimeter/length
approxPolyDP|contours|Curve,epsilon,closed flag|Simplified polygon
boundingRect|contours|Contour or nonzero mask|x,y,width,height
isContourConvex|contours|Ordered contour|Convexity Boolean
getRotationMatrix2D|affine|center(x,y),degrees,uniform scale|2x3 rotation/scale transform
getAffineTransform|affine|Three noncollinear point pairs|2x3 affine matrix
warpAffine|affine|2x3 matrix and output(width,height)|Resampled image
getPerspectiveTransform|perspective|Four ordered nondegenerate point pairs|3x3 homography
findHomography|ransac|Corresponding2D points,method,threshold|Homography and inlier mask; may fail
warpPerspective|perspective|3x3 matrix and output(width,height)|Projectively resampled image
HOGDescriptor|hog|Window/block/stride/cell sizes and bins|Descriptor object; compute on matching window
getGaborKernel|gabor|Kernel size,sigma,theta,wavelength,aspect,phase|Oriented sinusoidal Gaussian kernel
HoughLines|hough|Binary edges,rho/theta resolution,vote threshold|Polar line candidates or None
HoughLinesP|hough|Binary edges,resolution,votes,minLength,maxGap|Segment endpoint candidates or None
HoughCircles|circles|Grayscale and method-dependent parameters|Circle candidates or None
SIFT_create|sift|Optional detector parameters|SIFT feature object
ORB_create|orb|Optional detector parameters|ORB binary feature object
drawKeypoints|sift|Image,keypoint list,output or None,flags|Keypoint visualization
BFMatcher|matching|Norm type and optional crossCheck|Descriptor matcher
VideoCapture|video|Local file or camera index|Frame source with isOpened/read/release
VideoWriter|video|Path,codec,fps,(width,height)|Frame sink with write/release
undistort|calibration|Image,calibrated camera matrix,distortion coefficients|Undistorted image
'''
import supplement_topics
_have={r.split("|")[0] for r in data.strip().splitlines()}
data+=''.join(r+'\n' for r in supplement_topics.API_ROWS.strip().splitlines() if r.split('|')[0] not in _have)
apis=[]
for row in data.strip().splitlines():
    name,topic,inputs,outputs=row.split('|')
    obj=getattr(cv,name)
    # Short local API signature only, not copying reference documentation bodies.
    doclines=(getattr(obj,'__doc__','') or '').splitlines()
    signature=doclines[0] if doclines else name+'(...) [binding does not expose a signature; see input notes]'
    apis.append(dict(id='api:cv2.'+name,title='cv2.'+name,aliases=['cv.'+name,name],
        signature=signature[:500],input_notes=inputs,output_notes=outputs,topic_id=topic,
        source_refs=['topic:'+topic],origin='Installed OpenCV binding plus authored notes',
        verified_symbol_exists=True,version=cv.__version__,scope='core'))
for module,names in [(np,['sqrt','hypot','arctan2','degrees','histogram','zeros','ones','clip','reshape','bincount','mean','std','quantile','log1p']),
                     (plt,['imshow','subplot','subplots','bar','hist','plot','title','axis','savefig','close','tight_layout'])]:
    for name in names:
        obj=getattr(module,name)
        try: signature=str(inspect.signature(obj))
        except (ValueError,TypeError): signature=(obj.__doc__ or '').splitlines()[0]
        prefix='numpy' if module is np else 'matplotlib.pyplot'
        alias='np' if module is np else 'plt'
        apis.append(dict(id='api:'+prefix+'.'+name,title=prefix+'.'+name,aliases=[alias+'.'+name,name],signature=signature[:500],
            input_notes='See topic numpy/histogram or display; these are library signatures, not generated code.',
            output_notes='Return type varies by function; consult installed help for full overloads.',
            source_refs=['topic:numpy' if module is np else 'topic:display'],
            origin='Installed Python callable signature',verified_symbol_exists=True,scope='core',
            version=np.__version__ if module is np else matplotlib.__version__))
for title,obj,inputs,outputs,topic in [
    ('cv2.SIFT.detectAndCompute',cv.SIFT_create().detectAndCompute,'Grayscale image and optional mask','Keypoint list and descriptor matrix or None','sift'),
    ('cv2.BFMatcher.knnMatch',cv.BFMatcher().knnMatch,'Compatible descriptors and neighbor count','Lists of up to k DMatch objects','matching')]:
    apis.append(dict(id='api:'+title,title=title,aliases=[title.split('.')[-1]],signature=obj.__doc__.splitlines()[0],
        input_notes=inputs,output_notes=outputs,topic_id=topic,source_refs=['topic:'+topic],origin='Installed OpenCV binding',
        verified_symbol_exists=True,version=cv.__version__,scope='core'))
(ROOT/'records'/'apis.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in apis),encoding='utf-8')
print(json.dumps({'apis':len(apis),'opencv':cv.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}))
