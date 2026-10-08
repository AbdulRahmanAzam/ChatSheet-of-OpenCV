"""Lab 06 teaching prototypes. Actual camera calibration and datasets are required
for deployment. Missing/unauthorized/on-off states need explicit reference policies.
"""
import cv2 as cv
import numpy as np
from cv_core import gray,hough_segments,sift_match,stitch_with_homography,haar_denoise,process_video


def task01_screens(image,expected_rectangles,edge_support=0.35,dark_threshold=35):
    """Front-facing, calibrated screen ROIs [(x,y,w,h),...]. Hough-supported sides.
    Output is visible/dark/bright or unconfirmed; 'missing' cannot be proven from
    absent lines alone (occlusion/glare also remove edges). No inferred power state.
    """
    edges,lines=hough_segments(image); h,w=edges.shape
    line_canvas=np.zeros_like(edges)
    for x1,y1,x2,y2 in lines: cv.line(line_canvas,(x1,y1),(x2,y2),255,2)
    # Tolerance band accounts for double edges and small calibration errors.
    support=cv.dilate(line_canvas,np.ones((7,7),np.uint8))
    annotated=image.copy(); results=[]
    for index,(x,y,rw,rh) in enumerate(expected_rectangles):
        x,y,rw,rh=map(int,(x,y,rw,rh))
        if rw<12 or rh<12 or x<0 or y<0 or x+rw>w or y+rh>h: raise ValueError('Invalid calibrated ROI')
        x2,y2=x+rw-1,y+rh-1
        side_support=[np.mean(support[y,x:x+rw]>0),np.mean(support[y2,x:x+rw]>0),
                      np.mean(support[y:y+rh,x]>0),np.mean(support[y:y+rh,x2]>0)]
        visible=sum(s>=edge_support for s in side_support)>=3
        brightness=float(np.mean(gray(image)[y+5:y2-4,x+5:x2-4]))
        state=('visible_dark' if brightness<dark_threshold else 'visible_bright') if visible else 'unconfirmed_possible_missing_or_occluded'
        results.append({'screen':index,'state':state,'side_support':list(map(float,side_support)),
                        'mean_interior_intensity':brightness,'expected_xywh':[x,y,rw,rh]})
        color=(0,255,0) if visible else (0,0,255)
        cv.rectangle(annotated,(x,y),(x2,y2),color,2)
        cv.putText(annotated,f'{index}: '+('dark' if visible and brightness<dark_threshold else 'bright' if visible else 'check'),
                   (x,y-5 if y>15 else y+15),cv.FONT_HERSHEY_SIMPLEX,.4,color,1)
    return {'gray':gray(image),'edges':edges,'lines_on_black':line_canvas,'overlay':annotated,'screens':results}


def task02_inventory(scene,references):
    """references maps asset label->reference image. Similar-looking items may be ambiguous."""
    return {name:sift_match(ref,scene) for name,ref in references.items()}


def task03_wavelet_anomalies(signal,clean_baseline,levels=3,mad_factor=5):
    baseline=np.asarray(clean_baseline,float); signal=np.asarray(signal,float)
    _,threshold=haar_denoise(baseline,levels)
    # Score the detail component against an approximation-only reconstruction.
    # A soft-threshold-denoising residual is capped by its threshold and can hide
    # a large retained impulse, so it must not be the anomaly statistic here.
    baseline_coarse,_=haar_denoise(baseline,levels,float('inf'))
    residual=baseline-baseline_coarse; center=float(np.median(residual))
    sigma=float(np.median(np.abs(residual-center))/0.67448975)
    cutoff=max(mad_factor*sigma,1e-8)
    denoised,_=haar_denoise(signal,levels,threshold)
    coarse,_=haar_denoise(signal,levels,float('inf'))
    detail=signal-coarse
    score=np.abs(detail-center)
    return {'denoised':denoised,'residual':detail,'score':score,
            'anomaly_indices':np.flatnonzero(score>cutoff),'cutoff':cutoff,'wavelet_threshold':threshold}


def task04_recognize(reference,scene):
    result=sift_match(reference,scene); out=scene.copy()
    if result['found']: cv.polylines(out,[np.int32(np.rint(result['quad']))],True,(0,255,0),3)
    return result,out


def task04_video(reference,path,output):
    return process_video(path,lambda frame:task04_recognize(reference,frame)[1],output)


def task05_panorama(images):
    if len(images)<2: raise ValueError('At least two overlapping images required')
    result=images[0].copy(); steps=[]
    for im in images[1:]:
        match=sift_match(im,result)
        if not match['found']: raise ValueError('Cannot align panorama: '+match['reason'])
        result,_=stitch_with_homography(result,im,match['homography'])
        steps.append({'matches':match['matches'],'inliers':match['inliers']})
    return result,steps


def task06_lanes(image):
    edges,_=hough_segments(image); h,w=edges.shape
    roi=np.zeros_like(edges)
    polygon=np.int32([[(0,h-1),(int(w*.43),int(h*.55)),(int(w*.57),int(h*.55)),(w-1,h-1)]])
    cv.fillPoly(roi,polygon,255); cropped=cv.bitwise_and(edges,roi)
    raw=cv.HoughLinesP(cropped,1,np.pi/180,25,minLineLength=30,maxLineGap=25)
    groups={'left':[],'right':[]}; output=image.copy(); fits={}
    if raw is not None:
        for x1,y1,x2,y2 in raw.reshape(-1,4):
            if abs(x2-x1)<2: continue
            slope=(y2-y1)/(x2-x1)
            if not .35<abs(slope)<5: continue
            if slope<0 and (x1+x2)/2<w*.6: groups['left'] += [(x1,y1),(x2,y2)]
            elif slope>0 and (x1+x2)/2>w*.4: groups['right'] += [(x1,y1),(x2,y2)]
    for name,points in groups.items():
        if len(points)<2: continue
        p=np.array(points); a,b=np.polyfit(p[:,1],p[:,0],1) # x=a*y+b avoids division by slope
        ys=np.array([h-1,int(.6*h)]); xs=np.clip(a*ys+b,0,w-1).astype(int)
        cv.line(output,(int(xs[0]),int(ys[0])),(int(xs[1]),int(ys[1])),(0,255,0),5)
        fits[name]=list(zip(map(int,xs),map(int,ys)))
    return {'edges':edges,'ROI edges':cropped,'overlay':output,'lanes':fits}


def task07_coins(image,min_radius=10,max_radius=80,min_distance=30,param2=25):
    g=cv.medianBlur(gray(image),5)
    circles=cv.HoughCircles(g,cv.HOUGH_GRADIENT,dp=1.2,minDist=min_distance,
                            param1=120,param2=param2,minRadius=min_radius,maxRadius=max_radius)
    found=np.empty((0,3),dtype=int) if circles is None else np.rint(circles[0]).astype(int)
    output=image.copy()
    for x,y,r in found:
        cv.circle(output,(x,y),r,(0,255,0),2); cv.circle(output,(x,y),2,(0,0,255),-1)
    return {'blurred':g,'circles':found,'count':len(found),'overlay':output}


class ZoneMonitor:
    """Task 8. Policy: new foreground in calibrated zone for N consecutive frames.
    Static camera and empty reference frame required. Produces local events only.
    """
    def __init__(self,background,polygon,persistence=3,min_area=100,threshold=30):
        self.background=cv.GaussianBlur(gray(background),(5,5),0)
        self.zone=np.zeros_like(self.background); cv.fillPoly(self.zone,[np.int32(polygon)],255)
        if persistence<1 or min_area<=0 or threshold<0: raise ValueError('Invalid monitor settings')
        self.persistence=persistence; self.min_area=min_area; self.threshold=threshold
        self.count=0; self.active=False
    def update(self,frame):
        g=cv.GaussianBlur(gray(frame),(5,5),0)
        if g.shape!=self.background.shape: raise ValueError('Frame shape changed')
        diff=cv.absdiff(g,self.background)
        mask=cv.threshold(diff,self.threshold,255,cv.THRESH_BINARY)[1]
        mask=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask=cv.bitwise_and(mask,self.zone)
        contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        boxes=[cv.boundingRect(c) for c in contours if cv.contourArea(c)>=self.min_area]
        self.count=self.count+1 if boxes else 0
        active=self.count>=self.persistence; event=active and not self.active; self.active=active
        out=frame.copy()
        for x,y,w,h in boxes: cv.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
        return {'mask':mask,'boxes':boxes,'consecutive_frames':self.count,'active':active,
                'new_event':event,'policy':'new foreground in restricted zone','overlay':out}
