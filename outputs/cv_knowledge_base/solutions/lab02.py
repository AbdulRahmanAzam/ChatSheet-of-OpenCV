"""Educational visualization exercises. Intensity colors do not diagnose disease
or measure blood flow. CT/MRI arrays must already be registered for fusion.
"""
import cv2 as cv
import numpy as np
from cv_core import gray, u8, gamma_correct, log_correct, process_video


def task01_xray(image, dense_threshold=180, gamma=0.6, bgr_gains=(0.9,1.0,1.1)):
    g=gray(image); equalized=cv.equalizeHist(g)
    heat=cv.applyColorMap(equalized,cv.COLORMAP_JET)
    balance=u8(heat.astype(float)*np.asarray(bgr_gains))
    _,dense=cv.threshold(equalized,dense_threshold,255,cv.THRESH_BINARY)
    return {'Original gray':g,'Equalized':equalized,'JET intensity':heat,
            'Simulated channel gains':balance,'High intensity mask':dense,
            'Log':log_correct(equalized),'Gamma':gamma_correct(equalized,gamma)}


def task02_fusion(ct,mri, *, registered, alpha=0.7, gamma=0.6):
    if not registered: raise ValueError('Supply registered corresponding slices; resizing is not registration')
    if ct.shape[:2]!=mri.shape[:2]: raise ValueError('Registered arrays must share shape')
    if not 0<=alpha<=1: raise ValueError('alpha must be in [0,1]')
    a=cv.equalizeHist(gray(ct)); b=cv.equalizeHist(gray(mri))
    ca=cv.applyColorMap(a,cv.COLORMAP_JET); cb=cv.applyColorMap(b,cv.COLORMAP_HOT)
    merged=cv.addWeighted(ca,alpha,cb,1-alpha,0)
    return {'CT equalized':a,'MRI equalized':b,'CT JET':ca,'MRI HOT':cb,
            'CT weighted fusion':merged,'Log fusion':log_correct(merged),'Gamma fusion':gamma_correct(merged,gamma)}


def task03_echo_frame(frame):
    stages=task01_xray(frame)
    return np.hstack([frame,stages['Simulated channel gains'],
                      cv.cvtColor(stages['Log'],cv.COLOR_GRAY2BGR),
                      cv.cvtColor(stages['Gamma'],cv.COLOR_GRAY2BGR)])


def task03_echo_video(path,output_path):
    return process_video(path,task03_echo_frame,output_path)
