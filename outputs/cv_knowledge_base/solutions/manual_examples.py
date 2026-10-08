"""Corrected reconstructions of every identified manual code example group.
These are original runnable equivalents, NOT verified verbatim transcriptions.
Exact source code remains visible in page scans; OCR is preserved separately.
Functions accept a decoded BGR uint8 image; imports/load/display are centralized.
"""
import cv2 as cv
import numpy as np
from cv_core import (gray,gradients,direction_histogram,texture_stats,hog_features,uniform_lbp,
                     convolution,region_grow,watershed_stages,segment_kmeans,grid)


def basic_examples(image,second_image=None):
    """Lab01 physical pp14-24: reading is cv_core.read_image; plots use grid."""
    g=gray(image); h,w=g.shape
    text=image.copy()
    cv.putText(text,'Domain Expansion',(20,min(h-1,60)),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),3,cv.LINE_AA)
    M=cv.getRotationMatrix2D((w/2,h/2),45,1)
    examples={'color':image,'gray':g,'half_size':cv.resize(image,None,fx=.5,fy=.5),
              'gaussian31':cv.GaussianBlur(image,(31,31),0),'left_crop':image[:,:w//2],
              'text':text,'threshold100_max200':cv.threshold(g,100,200,cv.THRESH_BINARY)[1],
              'rotate_same_canvas':cv.warpAffine(image,M,(w,h)), 'equalized':cv.equalizeHist(g)}
    if second_image is not None:
        other=cv.resize(second_image,(w,h))
        examples['saturating_add']=cv.add(image,other)
        examples['alpha_blend']=cv.addWeighted(image,.5,other,.5,0)
    return examples


def feature_examples(image):
    """Lab04 pp7-15: HOG, LBP, HED, HIG, texture energy and contrast.
    OpenCV HOG and circular NumPy/OpenCV LBP replace optional scikit-image examples;
    their exact numerical descriptors should not be mixed with skimage-trained data.
    """
    lbp,lh=uniform_lbp(image); energy,contrast=texture_stats(image,3)
    hed,bins=direction_histogram(image,8,edge_only=True,signed=True,weighted=False)
    hig,hig_bins=direction_histogram(image,9,edge_only=False,signed=False,weighted=True)
    return {'hog':hog_features(image),'lbp_image':lbp,'lbp_histogram':lh,
            'hed':hed,'hed_bins':bins,'hig':hig,'hig_bins':hig_bins,'energy':energy,'contrast':contrast}


def filtering_examples(image):
    """Lab04 pp16-19 plus user-requested median/bilateral/depth/borders."""
    g=gray(image); f=g.astype(np.float32)
    kernel=np.float32([[1,0,-1],[2,0,-2],[1,0,-1]])
    gaussian=cv.getGaussianKernel(5,1)
    emboss=np.float32([[-2,-1,0],[-1,1,1],[0,1,2]])
    return {'box':cv.filter2D(f,-1,np.ones((3,3),np.float32)/9),
            'gaussian2d':cv.filter2D(f,-1,gaussian@gaussian.T),
            'gaussian_separable':cv.sepFilter2D(f,-1,gaussian,gaussian),
            'median':cv.medianBlur(g,5),'bilateral':cv.bilateralFilter(image,7,50,50),
            'sobel_x':cv.Sobel(f,cv.CV_32F,1,0,ksize=3),
            'sobel_y':cv.Sobel(f,cv.CV_32F,0,1,ksize=3),
            'scharr_x':cv.Scharr(f,cv.CV_32F,1,0),'scharr_y':cv.Scharr(f,cv.CV_32F,0,1),
            'emboss_signed':cv.filter2D(f,cv.CV_32F,emboss),
            'correlation_same':cv.filter2D(f,cv.CV_32F,kernel),
            'convolution_full':convolution(f,kernel,'full'),
            'convolution_valid':convolution(f,kernel,'valid'),
            'convolution_same_zero':convolution(f,kernel,'same'),
            'convolution_stride2':convolution(f,kernel,'valid',2)}


def segmentation_examples(image,seed=(10,10),foreground='dark'):
    """Lab05 pp8-15, corrected display colors and explicit coordinate convention."""
    g=gray(image); hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    otsu,otsu_mask=cv.threshold(g,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    return {'global':cv.threshold(g,128,255,cv.THRESH_BINARY)[1],
            'adaptive':cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2),
            'otsu_threshold':otsu,'otsu':otsu_mask,
            'hsv_mask':cv.inRange(hsv,np.array([30,50,50]),np.array([60,255,255])),
            'canny':cv.Canny(cv.GaussianBlur(g,(5,5),1),100,200),
            'region_grow':region_grow(g,seed,50),
            'watershed':watershed_stages(image,.2,foreground),
            'kmeans':segment_kmeans(image,4)[0]}


def edge_feature_examples(image):
    """Lab06 pp9-10,12-13,15-16,22-23: Sobel, Canny, LoG, SIFT."""
    g=gray(image); dx,dy,magnitude,angle=gradients(g)
    blur=cv.GaussianBlur(g,(5,5),1.4)
    lap=cv.Laplacian(blur,cv.CV_32F)
    # Zero crossings require a sign change and a minimum local response range.
    maximum=cv.dilate(lap,np.ones((3,3),np.uint8))
    minimum=cv.erode(lap,np.ones((3,3),np.uint8))
    crossings=np.uint8((minimum<0)&(maximum>0)&((maximum-minimum)>10))*255
    keypoints,descriptors=cv.SIFT_create().detectAndCompute(g,None)
    drawn=cv.drawKeypoints(image,keypoints,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return {'sobel_x':dx,'sobel_y':dy,'magnitude':magnitude,'angle':angle,
            'canny':cv.Canny(blur,50,150),'laplacian_signed':lap,
            'laplacian_absolute_display':cv.convertScaleAbs(lap),'log_zero_crossings':crossings,
            'sift_keypoints_image':drawn,'keypoint_count':len(keypoints),
            'descriptor_shape':None if descriptors is None else list(descriptors.shape)}


def morphology_examples(mask):
    k=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    return {'erode':cv.erode(mask,k),'dilate':cv.dilate(mask,k),
            'opening':cv.morphologyEx(mask,cv.MORPH_OPEN,k),
            'closing':cv.morphologyEx(mask,cv.MORPH_CLOSE,k),
            'gradient':cv.morphologyEx(mask,cv.MORPH_GRADIENT,k),
            'top_hat':cv.morphologyEx(mask,cv.MORPH_TOPHAT,k),
            'black_hat':cv.morphologyEx(mask,cv.MORPH_BLACKHAT,k)}
