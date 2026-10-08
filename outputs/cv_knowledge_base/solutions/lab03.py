"""Lab 03. PDF physical order is tasks 9-10, 5-8, 1-4. Matrices map source to destination."""
import cv2 as cv
import numpy as np
from cv_core import translation,rotation,warp_fit,affine_from_three,perspective_quad,transform_points,stitch_with_homography


def task01_scale(image,factor=3.0):
    """Default means 'to 300%'. For literal 'increase by 300%' use factor=4."""
    if factor<=0: raise ValueError('factor must be positive')
    h,w=image.shape[:2]; target=(round(w*factor),round(h*factor))
    S=np.diag([factor,factor,1.])
    H=translation((target[0]-1)/2,(target[1]-1)/2)@S@translation(-(w-1)/2,-(h-1)/2)
    return cv.warpPerspective(image,H,target),H


def task02_undo_rotation(image,observed_clockwise_degrees=45):
    h,w=image.shape[:2]; cx,cy=(w-1)/2,(h-1)/2
    H=translation(cx,cy)@rotation(-observed_clockwise_degrees)@translation(-cx,-cy)
    return warp_fit(image,H)


def task03_deshear(image,observed_shear=0.3):
    H=np.array([[1.,-observed_shear,0],[0,1,0],[0,0,1]])
    return warp_fit(image,H)


def task04_translate(image,offset=(150,80),canvas=None):
    H=translation(*offset); h,w=image.shape[:2]
    if canvas is None: canvas=(w+max(0,int(offset[0])),h+max(0,int(offset[1])))
    return cv.warpPerspective(image,H,canvas),H


def task05_rigid(image,angle=20,offset=(50,30),canvas=None):
    H=translation(*offset)@rotation(angle)
    h,w=image.shape[:2]
    if canvas is None: canvas=(2*w,2*h)
    return cv.warpPerspective(image,H,canvas),H


def task06_similarity(image,scale=1.3,angle=20,offset=(50,30),canvas=None):
    if scale<=0: raise ValueError('scale must be positive')
    H=translation(*offset)@rotation(angle)@np.diag([scale,scale,1])
    h,w=image.shape[:2]
    return cv.warpPerspective(image,H,canvas or (2*w,2*h)),H


def task07_affine(image,src_points,dst_points,canvas=None):
    H=affine_from_three(src_points,dst_points); h,w=image.shape[:2]
    return cv.warpPerspective(image,H,canvas or (w,h)),H


def task08_rectify(image,corners,size=(400,400)):
    return perspective_quad(image,corners,size)


def task09_panorama(left,right,left_points,right_points):
    a,b=np.asarray(left_points,np.float32),np.asarray(right_points,np.float32)
    if a.shape != b.shape or a.ndim!=2 or a.shape[1]!=2 or len(a)<4:
        raise ValueError('At least four paired points required')
    H,inliers=cv.findHomography(b,a,cv.RANSAC,3.0)
    if H is None or inliers is None or inliers.sum()<4: raise ValueError('Degenerate correspondences')
    result,T=stitch_with_homography(left,right,H)
    return result,H,T


def task10_painting(painting,wall,destination_quad,scale=0.8,angle=10,offset=(30,20)):
    if scale<=0: raise ValueError('scale must be positive')
    h,w=painting.shape[:2]; corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    linear=np.diag([scale,scale,1.]); rigid=translation(*offset)@rotation(angle)
    moved=transform_points(corners,rigid@linear).astype(np.float32)
    dst=np.asarray(destination_quad,np.float32)
    if dst.shape!=(4,2) or not cv.isContourConvex(dst) or abs(cv.contourArea(dst))<1:
        raise ValueError('Destination must be an ordered convex quadrilateral')
    projective=cv.getPerspectiveTransform(moved,dst)
    H=projective@rigid@linear
    wh=(wall.shape[1],wall.shape[0])
    warped=cv.warpPerspective(painting,H,wh)
    mask=cv.warpPerspective(np.full((h,w),255,np.uint8),H,wh,flags=cv.INTER_NEAREST)
    out=wall.copy(); out[mask>0]=warped[mask>0]
    return {'image':out,'linear':linear,'rigid':rigid,'projective':projective,'composite':H}
