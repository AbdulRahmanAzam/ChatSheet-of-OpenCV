"""Original, corrected teaching implementations. No network calls or model weights.
All images are BGR uint8 unless stated; points are (x,y), arrays are [y,x].
See TASK_SOLUTIONS.md for assumptions and source-page references.
"""
from collections import deque
from pathlib import Path
import cv2 as cv
import numpy as np


def read_image(path, grayscale=False):
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return image


def gray(image):
    return image.copy() if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def u8(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def grid(images, path, columns=3):
    """Save a mapping title->image; display 3-channel inputs as BGR, masks as gray."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(images)
    fig, axes = plt.subplots((n+columns-1)//columns, columns,
                             figsize=(5*columns, 4*((n+columns-1)//columns)), squeeze=False)
    for ax, (title, im) in zip(axes.flat, images.items()):
        if im.ndim == 3:
            ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        elif im.dtype == np.uint8:
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(im, cmap='viridis')
        ax.set_title(title); ax.axis('off')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def gamma_correct(image, gamma=0.6):
    """Convention: output = 255*(input/255)**gamma. gamma<1 brightens."""
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    return cv.LUT(image, u8(255*(np.arange(256)/255.)**gamma))


def log_correct(image):
    return u8(255*np.log1p(image.astype(np.float32))/np.log(256))


def gradients(image):
    g = gray(image)
    dx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    dy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    return dx, dy, np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))


def direction_histogram(image, bins=9, edge_only=False, signed=False, weighted=True):
    g = cv.GaussianBlur(gray(image), (5, 5), 1)
    dx, dy, mag, angle = gradients(g)
    period = 360 if signed else 180
    select = cv.Canny(g, 30, 70) > 0 if edge_only else mag > 0
    hist, edges = np.histogram(angle[select] % period, bins=bins, range=(0, period),
                               weights=mag[select] if weighted else None)
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1e-12), edges


def texture_stats(image, size=7):
    f = gray(image).astype(np.float32)/255
    mean = cv.boxFilter(f, -1, (size, size), normalize=True)
    second = cv.boxFilter(f*f, -1, (size, size), normalize=True)
    return second*size*size, np.sqrt(np.maximum(second-mean*mean, 0))


def uniform_lbp(image):
    """Circular P=8,R=1 bilinear LBP; uniform bins 0..8, nonuniform 9.
    Excludes a 1-pixel border. Self-contained alternative, not promised bit-identical
    to scikit-image (sampling/rounding and border choices differ).
    """
    f = gray(image).astype(np.float32)
    if min(f.shape) < 3:
        raise ValueError('LBP requires at least 3x3')
    yy, xx = np.mgrid[1:f.shape[0]-1, 1:f.shape[1]-1].astype(np.float32)
    center = f[1:-1, 1:-1]
    bits = []
    for p in range(8):
        a = 2*np.pi*p/8
        neighbor = cv.remap(f, xx+np.float32(np.cos(a)), yy-np.float32(np.sin(a)), cv.INTER_LINEAR)
        bits.append(neighbor >= center)
    bits = np.stack(bits).astype(np.uint8)
    transitions = np.sum(bits != np.roll(bits, 1, axis=0), axis=0)
    labels = np.where(transitions <= 2, bits.sum(axis=0), 9).astype(np.uint8)
    hist = np.bincount(labels.ravel(), minlength=10).astype(np.float32)
    return labels, hist/hist.sum()


def hog_features(image):
    """OpenCV HOG on a fixed 64x64 grayscale window; 8x8 cells, 2x2-cell blocks."""
    g = cv.resize(gray(image), (64, 64), interpolation=cv.INTER_AREA)
    hog = cv.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    return hog.compute(g).ravel()


def convolution(image, kernel, mode='same', stride=1):
    """True 2-D convolution, odd kernel, float32 output, explicit zero padding.
    filter2D itself correlates; flipping both axes turns it into convolution.
    """
    f = np.asarray(image, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim != 2 or any(n % 2 == 0 for n in k.shape) or stride < 1:
        raise ValueError('Use an odd 2-D kernel and positive integer stride')
    ry, rx = k.shape[0]//2, k.shape[1]//2
    if mode == 'full':
        f = cv.copyMakeBorder(f, ry, ry, rx, rx, cv.BORDER_CONSTANT, value=0)
    elif mode not in ('same', 'valid'):
        raise ValueError('mode must be same, valid or full')
    result = cv.filter2D(f, cv.CV_32F, k[::-1, ::-1], borderType=cv.BORDER_CONSTANT)
    if mode == 'valid':
        if f.shape[0] < k.shape[0] or f.shape[1] < k.shape[1]:
            raise ValueError('valid kernel exceeds image')
        result = result[ry:result.shape[0]-ry if ry else None,
                        rx:result.shape[1]-rx if rx else None]
    return result[::stride, ::stride]


def translation(tx, ty):
    return np.array([[1.,0,tx], [0,1.,ty], [0,0,1.]])


def rotation(degrees):
    """Cartesian matrix on image coordinates: positive appears clockwise (y down)."""
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0], [s,c,0], [0,0,1.]])


def transform_points(points, matrix):
    p = np.c_[np.asarray(points, dtype=float), np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(np.abs(p[:,2]) < 1e-10):
        raise ValueError('Point maps to infinity')
    return p[:,:2]/p[:,2:]


def warp_fit(image, matrix, max_pixels=40_000_000):
    """Expand output to transformed pixel-center bounds; return image and effective H."""
    h,w = image.shape[:2]
    corners = transform_points([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], matrix)
    lo = np.floor(corners.min(axis=0)+1e-7); hi = np.ceil(corners.max(axis=0)-1e-7)
    width,height = (hi-lo+1).astype(int)
    if width < 1 or height < 1 or width*height > max_pixels:
        raise ValueError('Implausible transform/output canvas')
    H = translation(-lo[0], -lo[1]) @ matrix
    return cv.warpPerspective(image, H, (width,height)), H


def affine_from_three(src, dst):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != (3,2) or dst.shape != (3,2):
        raise ValueError('Exactly 3 corresponding 2D points required')
    a = np.zeros((6,6)); b = dst.reshape(-1)
    a[::2,:3] = np.c_[src,np.ones(3)]; a[1::2,3:] = np.c_[src,np.ones(3)]
    if np.linalg.matrix_rank(a) < 6:
        raise ValueError('Source points are collinear')
    m = np.linalg.solve(a,b).reshape(2,3)
    return np.vstack([m,[0,0,1]])


def perspective_quad(image, src, size=(400,400)):
    """src order: top-left, top-right, bottom-right, bottom-left."""
    src = np.asarray(src, np.float32)
    if src.shape != (4,2) or abs(cv.contourArea(src)) < 1 or not cv.isContourConvex(src):
        raise ValueError('Four ordered, convex, nondegenerate points required')
    w,h = size
    dst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    H = cv.getPerspectiveTransform(src,dst)
    return cv.warpPerspective(image,H,(w,h)), H


def stitch_with_homography(left, right, right_to_left, max_pixels=40_000_000):
    """Warp masks as well as images so black pixels remain valid; average overlap."""
    hl,wl = left.shape[:2]; hr,wr = right.shape[:2]
    corners = np.vstack([[[0,0],[wl-1,0],[wl-1,hl-1],[0,hl-1]],
                         transform_points([[0,0],[wr-1,0],[wr-1,hr-1],[0,hr-1]],right_to_left)])
    # Do not add a spurious border for numerical noise around integer coordinates.
    lo = np.floor(corners.min(0)+1e-7); hi = np.ceil(corners.max(0)-1e-7)
    w,h = (hi-lo+1).astype(int)
    if w*h > max_pixels or min(w,h) < 1:
        raise ValueError('Implausible panorama canvas')
    T = translation(*(-lo))
    a = cv.warpPerspective(left,T,(w,h)).astype(np.float32)
    b = cv.warpPerspective(right,T@right_to_left,(w,h)).astype(np.float32)
    ma = cv.warpPerspective(np.ones((hl,wl),np.uint8),T,(w,h),flags=cv.INTER_NEAREST)
    mb = cv.warpPerspective(np.ones((hr,wr),np.uint8),T@right_to_left,(w,h),flags=cv.INTER_NEAREST)
    weight = (ma+mb).astype(float)
    if a.ndim == 3: weight = weight[...,None]
    return u8((a+b)/np.maximum(weight,1)), T


def region_grow(image, seed, tolerance=15, connectivity=4):
    """Fixed-seed intensity rule, NOT moving-mean region growth. Seed is (x,y)."""
    g = gray(image); h,w = g.shape; x,y = map(int,seed)
    if not (0 <= x < w and 0 <= y < h) or tolerance < 0 or connectivity not in (4,8):
        raise ValueError('Invalid seed, tolerance, or connectivity')
    ref = int(g[y,x]); seen = np.zeros((h,w),bool); mask = np.zeros((h,w),np.uint8)
    q = deque([(x,y)]); seen[y,x] = True
    offsets = [(1,0),(-1,0),(0,1),(0,-1)]
    if connectivity == 8: offsets += [(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        x,y = q.popleft()
        if abs(int(g[y,x])-ref) > tolerance: continue
        mask[y,x] = 255
        for dx,dy in offsets:
            xx,yy = x+dx,y+dy
            if 0 <= xx < w and 0 <= yy < h and not seen[yy,xx]:
                seen[yy,xx] = True; q.append((xx,yy))
    return mask


def watershed_stages(image, distance_fraction=0.5, foreground='bright'):
    if not 0 < distance_fraction < 1 or foreground not in ('bright','dark'):
        raise ValueError('fraction in (0,1), foreground bright or dark')
    g = gray(image)
    flag = cv.THRESH_BINARY if foreground == 'bright' else cv.THRESH_BINARY_INV
    threshold, binary = cv.threshold(g,0,255,flag|cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=2)
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    distance = cv.distanceTransform(opening,cv.DIST_L2,5)
    sure_fg = np.uint8(distance > distance_fraction*distance.max())*255
    unknown = cv.subtract(sure_bg,sure_fg)
    n, labels = cv.connectedComponents(sure_fg)
    markers = labels.astype(np.int32)+1; markers[unknown>0] = 0
    initial = markers.copy()
    bgr = image.copy() if image.ndim == 3 else cv.cvtColor(g,cv.COLOR_GRAY2BGR)
    final = cv.watershed(bgr,markers)
    overlay = bgr.copy(); overlay[final==-1] = (0,0,255)
    return {'gray':g,'binary':binary,'opening':opening,'sure_background':sure_bg,
            'distance':distance,'sure_foreground':sure_fg,'unknown':unknown,
            'initial_markers':initial,'markers':final,'overlay':overlay,
            'otsu_threshold':float(threshold),'seed_count':n-1,
            'region_count':len(set(np.unique(final))-{0,1,-1})}


def segment_kmeans(image, k=4, seed=42):
    pixels = image.reshape(-1,image.shape[-1]).astype(np.float32)
    if not 1 <= k <= len(pixels): raise ValueError('Invalid k')
    cv.setRNGSeed(seed)
    compactness,labels,centers = cv.kmeans(pixels,k,None,
        (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER,100,0.2),3,cv.KMEANS_PP_CENTERS)
    return u8(centers)[labels.ravel()].reshape(image.shape), labels.reshape(image.shape[:2]), float(compactness)


def sift_match(reference, scene, ratio=0.75, min_inliers=8):
    """Reference->scene homography; failure returned explicitly, never invented."""
    sift = cv.SIFT_create()
    ka,da = sift.detectAndCompute(gray(reference),None)
    kb,db = sift.detectAndCompute(gray(scene),None)
    result = {'found':False,'matches':0,'inliers':0,'reason':'insufficient descriptors'}
    if da is None or db is None or len(db)<2: return result
    pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(da,db,k=2)
    good = [pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < ratio*pair[1].distance]
    result['matches'] = len(good)
    if len(good)<4: result['reason']='fewer than four ratio-test matches'; return result
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    H,mask = cv.findHomography(src,dst,cv.RANSAC,4.0)
    if H is None or mask is None: result['reason']='homography failed'; return result
    count = int(mask.sum()); result['inliers']=count
    if count < min_inliers: result['reason']='insufficient geometric inliers'; return result
    h,w=reference.shape[:2]
    try: quad = transform_points([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],H).astype(np.float32)
    except ValueError: result['reason']='degenerate projection'; return result
    sh,sw=scene.shape[:2]
    area = abs(cv.contourArea(quad))
    if not np.isfinite(quad).all() or not cv.isContourConvex(quad) or not (25<area<4*sh*sw):
        result['reason']='implausible projected boundary'; return result
    result.update(found=True,reason='ratio test and RANSAC passed',homography=H,quad=quad)
    return result


def haar_denoise(signal, levels=3, threshold=None):
    """Orthonormal Haar DWT, soft detail thresholding, exact inverse and edge padding."""
    x = np.asarray(signal,dtype=float).ravel()
    if len(x)<2 or not np.isfinite(x).all() or levels<1: raise ValueError('Invalid signal')
    multiple = 2**levels
    approx = np.pad(x,(0,(-len(x))%multiple),mode='edge')
    details=[]
    for _ in range(levels):
        details.append((approx[::2]-approx[1::2])/np.sqrt(2))
        approx=(approx[::2]+approx[1::2])/np.sqrt(2)
    if threshold is None:
        sigma=np.median(np.abs(details[0]-np.median(details[0])))/0.67448975
        threshold=float(sigma*np.sqrt(2*np.log(len(x))))
    for detail in reversed(details):
        d=np.sign(detail)*np.maximum(np.abs(detail)-threshold,0)
        restored=np.empty(2*len(approx)); restored[::2]=(approx+d)/np.sqrt(2); restored[1::2]=(approx-d)/np.sqrt(2)
        approx=restored
    return approx[:len(x)], float(threshold)


def hough_segments(image):
    g = cv.GaussianBlur(gray(image),(5,5),1)
    edges = cv.Canny(g,50,150)
    lines = cv.HoughLinesP(edges,1,np.pi/180,40,minLineLength=40,maxLineGap=12)
    return edges, np.empty((0,4),dtype=int) if lines is None else lines.reshape(-1,4)


def process_video(path, frame_function, output_path, max_frames=None):
    """Offline video loop, no GUI. Returns processed frame count; writes silent MP4."""
    capture=cv.VideoCapture(str(path))
    if not capture.isOpened(): raise FileNotFoundError(f'Cannot open video: {path}')
    fps=capture.get(cv.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: fps=25
    writer=None; count=0
    try:
        while max_frames is None or count<max_frames:
            ok,frame=capture.read()
            if not ok: break
            rendered=frame_function(frame)
            if rendered.ndim==2: rendered=cv.cvtColor(rendered,cv.COLOR_GRAY2BGR)
            if writer is None:
                writer=cv.VideoWriter(str(output_path),cv.VideoWriter_fourcc(*'mp4v'),fps,
                                      (rendered.shape[1],rendered.shape[0]))
                if not writer.isOpened(): raise RuntimeError('MP4 encoder unavailable')
            writer.write(rendered); count+=1
    finally:
        capture.release()
        if writer is not None: writer.release()
    if not count: raise ValueError('Video contains no decodable frames')
    return count
