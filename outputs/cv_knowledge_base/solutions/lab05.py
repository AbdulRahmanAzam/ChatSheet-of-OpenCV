"""Segmentation experiments. Functions return actual masks and measurable outputs."""
import cv2 as cv
import numpy as np
from cv_core import gray,region_grow,watershed_stages,segment_kmeans


def task01_document(image):
    g=gray(image); out={'Original':g}
    for t in (80,127,180): out[f'Global {t}']=cv.threshold(g,t,255,cv.THRESH_BINARY_INV)[1]
    out['Adaptive Gaussian']=cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,31,7)
    return out


def task02_adaptive_sweep(image,block_sizes=(11,31,51),constants=(2,7,12)):
    g=gray(image); out={'Original':g}
    for method,name in [(cv.ADAPTIVE_THRESH_MEAN_C,'mean'),(cv.ADAPTIVE_THRESH_GAUSSIAN_C,'gaussian')]:
        for block in block_sizes:
            if block<=1 or block%2==0: raise ValueError('block size must be odd and greater than 1')
            for C in constants:
                out[f'{name}, block={block}, C={C}']=cv.adaptiveThreshold(g,255,method,cv.THRESH_BINARY,block,C)
    return out


def task03_otsu(image):
    g=gray(image); altered=cv.convertScaleAbs(g,alpha=1.2,beta=10)
    images={'Original':g,'Blurred':cv.GaussianBlur(g,(5,5),0),'Contrast changed':altered}
    thresholds={}; histograms={}
    for name,im in list(images.items()):
        t,m=cv.threshold(im,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
        thresholds[name]=float(t); histograms[name]=np.bincount(im.ravel(),minlength=256)
        images[name+' Otsu']=m
    return {'images':images,'thresholds':thresholds,'histograms':histograms}


def task04_yellow(image):
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    narrow=cv.inRange(hsv,np.array([22,120,100],np.uint8),np.array([32,255,255],np.uint8))
    broad=cv.inRange(hsv,np.array([15,60,50],np.uint8),np.array([40,255,255],np.uint8))
    return {'Original':image,'Narrow mask':narrow,'Narrow extraction':cv.bitwise_and(image,image,mask=narrow),
            'Broad mask':broad,'Broad extraction':cv.bitwise_and(image,image,mask=broad)}


def task05_canny(image):
    g=cv.GaussianBlur(gray(image),(5,5),1.2)
    out={'Smoothed':g}
    for low,high in [(30,60),(30,120),(30,200)]:
        out[f'Canny low={low}, high={high}']=cv.Canny(g,low,high,L2gradient=True)
    return out


def task06_region_grow(image,seeds,tolerances=(5,15,30)):
    if len(seeds)!=2: raise ValueError('Supply two (x,y) seeds')
    return {f'seed={seed}, tolerance={t}':region_grow(image,seed,t) for seed in seeds for t in tolerances}


def task07_watershed(image,foreground='bright'):
    return watershed_stages(image,0.5,foreground)


def task08_distance_sweep(image,foreground='bright'):
    return {str(f):watershed_stages(image,f,foreground) for f in (.2,.4,.6,.8)}


def task09_kmeans(image):
    result={}
    for k in (2,4,6):
        segmented,labels,compactness=segment_kmeans(image,k)
        result[k]={'image':segmented,'labels':labels,'compactness':compactness}
    return result


def task10_compare(image,truth=None):
    """Three dark-document segmentation methods; optional known text mask for IoU."""
    g=gray(image)
    masks={'Global127':cv.threshold(g,127,255,cv.THRESH_BINARY_INV)[1],
           'Otsu':cv.threshold(g,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)[1],
           'Adaptive':cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,31,7)}
    scores={}
    if truth is not None:
        if truth.shape!=g.shape: raise ValueError('Ground truth shape mismatch')
        target=truth>0
        for name,mask in masks.items():
            pred=mask>0; union=np.count_nonzero(pred|target)
            scores[name]=float(np.count_nonzero(pred&target)/union) if union else 1.0
    return {'masks':masks,'iou':scores,'best_on_supplied_truth':max(scores,key=scores.get) if scores else None}
