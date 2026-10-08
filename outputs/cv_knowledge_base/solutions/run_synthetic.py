"""Run offline demonstration and invariant checks: python solutions/run_synthetic.py.
Synthetic results are engineering checks, NOT evaluation on missing course data.
"""
from pathlib import Path
import json,sys,platform
import cv2 as cv
import numpy as np
import matplotlib
import cv_core as core
import lab01,lab02,lab03,lab04,lab05,lab06,manual_examples

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'demos'; OUT.mkdir(exist_ok=True)
checks=[]; exercised=[]
def check(name,condition):
    if not bool(condition): raise AssertionError(name)
    checks.append(name)
def rejects(name,function,error=ValueError):
    try: function()
    except error: checks.append(name); return
    raise AssertionError(name+' did not reject input')
def mark(lab,n): exercised.append(f'L{lab:02d}-T{n:02d}')
def save(name,images,columns=3): core.grid(images,OUT/(name+'.png'),columns)


def main():
    rng=np.random.default_rng(42); cv.setRNGSeed(42)
    # A distinctive synthetic target with edges and texture for geometric checks.
    image=rng.integers(0,256,(400,480,3),dtype=np.uint8)
    image=cv.GaussianBlur(image,(3,3),0)
    cv.rectangle(image,(30,35),(230,190),(20,210,240),-1)
    cv.circle(image,(340,230),65,(220,80,30),-1)
    cv.putText(image,'CV LAB',(55,300),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),4)
    cv.imwrite(str(OUT/'synthetic_input.png'),image)
    black=np.zeros_like(image)
    rejects('Unreadable image fails clearly',lambda:core.read_image(OUT/'not_present.png'),FileNotFoundError)
    gm=lab01.GroceryManager(); gm.add_item('apple',2,3); gm.add_item('apple',1,3)
    check('Grocery total and removal',gm.calculate_total()==9 and gm.remove_item('apple')['quantity']==3)
    rejects('Missing grocery raises KeyError',lambda:gm.remove_item('missing'),KeyError); mark(1,1)
    students={1:{'Name':'A','Major':'AI','Grades':[90,80]},2:{'Name':'B','Major':'CS','Grades':{'cv':70}}}
    check('Student average and major filter',lab01.task02_students(students,'ai')['top_names']==['A']); mark(1,2)
    read=lab01.task03_display(OUT/'synthetic_input.png',OUT/'lab01_display.png')
    check('Image decoded unchanged',np.array_equal(read,image)); mark(1,3)
    circles=lab01.task04_circles(); check('Concentric canvas dimensions',circles['image'].shape==(800,800,3)); mark(1,4)
    save('lab01_concentric',{'Five circles':circles['image']},1)
    rois=lab01.task05_blur_roi(image); check('Both center crops exactly 300',all(v.shape[:2]==(300,300) for v in rois.values())); mark(1,5)
    save('lab01_roi',rois,2)
    overlay=lab01.task06_overlay(image); check('Overlay leaves top 80 percent unchanged',np.array_equal(overlay[:320],image[:320])); mark(1,6)
    stages=lab01.task07_threshold_rotate(image); save('lab01_threshold_rotate',stages,2); mark(1,7)
    comp=lab01.task08_composite(image,black); check('Composite mask selects exact pixels',np.array_equal(comp['Composite'][comp['Mask']==0],np.zeros_like(comp['Composite'][comp['Mask']==0]))); mark(1,8)
    stats=lab01.task09_rgb_statistics(image); check('RGB stats count',stats['R']['count']==400*480); mark(1,9)
    medical=lab02.task01_xray(image); save('lab02_visualization_only',medical); mark(2,1)
    fused=lab02.task02_fusion(image,image,registered=True); check('Fusion uint8 range',fused['CT weighted fusion'].dtype==np.uint8); mark(2,2)
    rejects('Unregistered CT MRI rejected',lambda:lab02.task02_fusion(image,image,registered=False))
    # Test actual disk video round-trip, frame loop, encoder, and resource closure.
    video=OUT/'synthetic_input.avi'
    writer=cv.VideoWriter(str(video),cv.VideoWriter_fourcc(*'MJPG'),5,(480,400))
    check('Synthetic video encoder available',writer.isOpened())
    for _ in range(3): writer.write(image)
    writer.release()
    frames=lab02.task03_echo_video(video,OUT/'lab02_processed.mp4')
    check('All three video frames processed',frames==3); mark(2,3)
    scaled,H=lab03.task01_scale(image); check('300 percent scaling dimensions',scaled.shape[:2]==(1200,1440)); mark(3,1)
    rot,H=lab03.task02_undo_rotation(image); check('Rotation keeps four transformed corners in canvas',np.min(core.transform_points([[0,0],[479,0],[479,399],[0,399]],H))>=-1e-6); mark(3,2)
    sheared,H=lab03.task03_deshear(image); check('Shear has determinant one',np.isclose(np.linalg.det(H),1)); mark(3,3)
    shifted,H=lab03.task04_translate(image); check('Exact translation maps origin',np.allclose(core.transform_points([[0,0]],H),[[150,80]])); mark(3,4)
    rigid,H=lab03.task05_rigid(image); check('Rigid transform preserves point distances',np.isclose(np.linalg.norm(np.diff(core.transform_points([[20,30],[80,110]],H),axis=0)),100)); mark(3,5)
    similar,H=lab03.task06_similarity(image); check('Similarity uniformly scales distances',np.isclose(np.linalg.norm(np.diff(core.transform_points([[20,30],[80,110]],H),axis=0)),130)); mark(3,6)
    src=np.float32([[0,0],[479,0],[0,399]]); dst=src+np.float32([5,8])
    affine,H=lab03.task07_affine(image,src,dst); check('Solved affine matches landmarks',np.allclose(core.transform_points(src,H),dst)); mark(3,7)
    rejects('Collinear affine rejected',lambda:core.affine_from_three([[0,0],[1,1],[2,2]],[[0,0],[1,1],[2,2]]))
    quad=np.float32([[20,20],[450,30],[430,370],[35,350]])
    rectified,H=lab03.task08_rectify(image,quad); check('Perspective maps four corners',np.allclose(core.transform_points(quad,H),[[0,0],[399,0],[399,399],[0,399]],atol=1e-3)); mark(3,8)
    full=np.float32([[0,0],[479,0],[479,399],[0,399]])
    pano,_,_=lab03.task09_panorama(image,image,full,full); check('Identity panorama preserves pixels',np.array_equal(pano,image)); mark(3,9)
    painting=lab03.task10_painting(image,black,quad)
    check('Composed painting maps corners to destination',np.allclose(core.transform_points(full,painting['composite']),quad,atol=1e-3)); mark(3,10)
    save('lab03_geometry',{'Original':image,'Rectified':rectified,'Painting placement':painting['image'],'Rotation expanded':rot})
    features=lab04.task01_features(image)
    check('Texture features normalized and finite',np.isclose(features['lbp_histogram'].sum(),1) and np.isfinite(features['hog']).all())
    check('LBP identical exemplar distance zero',lab04.classify_material(image,[('synthetic',image)])['distance']==0); mark(4,1)
    _,contrast=core.texture_stats(np.full((30,30),255,np.uint8)); check('Uniform bright texture has zero contrast',np.max(contrast)==0)
    # Unevenly lit document with exact known text ground truth.
    truth=np.zeros((240,360),np.uint8)
    for y in [45,95,145,195]: cv.putText(truth,'VISION 123',(20,y),cv.FONT_HERSHEY_SIMPLEX,.85,255,2,cv.LINE_8)
    illumination=np.tile(np.linspace(80,240,360),(240,1))
    doc=core.u8(illumination*(1-.78*(truth>0)))
    docs=lab05.task01_document(doc); save('lab05_document',docs); mark(5,1)
    sweep=lab05.task02_adaptive_sweep(doc); check('All 18 adaptive combinations present',len(sweep)==19); save('lab05_adaptive_sweep',sweep,4); mark(5,2)
    coins=np.full((240,320,3),15,np.uint8)
    for center in [(105,125),(190,125)]: cv.circle(coins,center,52,(220,220,220),-1)
    otsu=lab05.task03_otsu(coins); check('Otsu splits synthetic foreground from background',otsu['images']['Original Otsu'][125,105]==255 and otsu['images']['Original Otsu'][0,0]==0); mark(5,3)
    save('lab05_otsu',otsu['images'])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for name,hist in otsu['histograms'].items(): ax.plot(np.arange(256),hist,label=name)
    ax.legend(); ax.set(xlabel='Intensity',ylabel='Pixel count',title='Synthetic coin intensity histograms')
    fig.savefig(OUT/'lab05_histograms.png'); plt.close(fig)
    yellow=lab05.task04_yellow(image); check('Narrow yellow mask is subset of broad mask',not np.any((yellow['Narrow mask']>0)&(yellow['Broad mask']==0))); mark(5,4)
    save('lab05_yellow',yellow)
    canny=lab05.task05_canny(image); check('Canny returns binary maps',all(set(np.unique(m)).issubset({0,255}) for n,m in canny.items() if n!='Smoothed')); mark(5,5)
    regions=lab05.task06_region_grow(coins,[(105,125),(0,0)]); check('Region growth respects intensity barrier',regions['seed=(105, 125), tolerance=5'][0,0]==0); mark(5,6)
    rejects('Region seed outside image rejected',lambda:core.region_grow(coins,(-1,2)))
    water=lab05.task07_watershed(coins); check('Watershed markers signed int32',water['markers'].dtype==np.int32 and np.any(water['markers']==-1)); mark(5,7)
    save('lab05_watershed',{k:v for k,v in water.items() if isinstance(v,np.ndarray)},4)
    dsweep=lab05.task08_distance_sweep(coins); check('Four distance experiments',len(dsweep)==4); mark(5,8)
    save('lab05_distance_sweep',{f"fraction {f}, seeds {r['seed_count']}, regions {r['region_count']}":r['overlay'] for f,r in dsweep.items()},2)
    small=cv.resize(image,(160,120)); clusters=lab05.task09_kmeans(small)
    check('K-means produces at most K colors',all(len(np.unique(r['image'].reshape(-1,3),axis=0))<=k for k,r in clusters.items())); mark(5,9)
    comparison=lab05.task10_compare(doc,truth); check('IoU scores bounded',all(0<=v<=1 for v in comparison['iou'].values())); mark(5,10)
    save('lab05_comparison',{'Known synthetic text':truth,**comparison['masks']},2)
    screen=np.full((300,500,3),80,np.uint8)
    cv.rectangle(screen,(30,60),(209,209),(240,240,240),3)
    cv.rectangle(screen,(35,65),(204,204),(10,10,10),-1)
    sr=lab06.task01_screens(screen,[(30,60,180,150),(280,60,180,150)])
    check('Reference screen visible and absent reference unconfirmed',sr['screens'][0]['state']=='visible_dark' and sr['screens'][1]['state'].startswith('unconfirmed')); mark(6,1)
    save('lab06_screen_reference',{k:v for k,v in sr.items() if isinstance(v,np.ndarray)},2)
    inventory=lab06.task02_inventory(image,{'target':image,'blank':black})
    check('SIFT identity match found and blank rejected',inventory['target']['found'] and not inventory['blank']['found']); mark(6,2)
    baseline=np.sin(np.linspace(0,8*np.pi,256))+rng.normal(0,.05,256)
    signal=baseline.copy(); signal[100]+=5
    anomaly=lab06.task03_wavelet_anomalies(signal,baseline)
    check('Haar zero threshold reconstructs odd length signal',np.allclose(core.haar_denoise(signal[:253],3,0)[0],signal[:253]))
    check('Synthetic impulse is actually flagged',100 in anomaly['anomaly_indices']); mark(6,3)
    fig,axes=plt.subplots(2,1,figsize=(10,6))
    axes[0].plot(signal,label='Signal with injected impulse'); axes[0].plot(anomaly['denoised'],label='Denoised')
    axes[0].legend(); axes[1].plot(anomaly['score'],label='Wavelet detail score')
    axes[1].axhline(anomaly['cutoff'],color='red',label='Clean-baseline cutoff'); axes[1].legend()
    axes[1].set_xlabel('Sample index'); fig.tight_layout(); fig.savefig(OUT/'lab06_wavelet.png'); plt.close(fig)
    found,annotated=lab06.task04_recognize(image,image); check('Reference recognition has geometric inliers',found['found'] and found['inliers']>=8); mark(6,4)
    recognition_frames=lab06.task04_video(image,video,OUT/'lab06_recognition.mp4'); check('Recognition video loop executes',recognition_frames==3)
    panorama,steps=lab06.task05_panorama([image,image]); check('Automatic panorama geometry verified',len(steps)==1 and steps[0]['inliers']>=8); mark(6,5)
    road=np.zeros((300,500,3),np.uint8)
    cv.line(road,(50,299),(225,165),(255,255,255),5); cv.line(road,(450,299),(275,165),(255,255,255),5)
    lanes=lab06.task06_lanes(road); check('Both synthetic lane sides fitted',set(lanes['lanes'])=={'left','right'}); mark(6,6)
    save('lab06_lanes',{k:v for k,v in lanes.items() if isinstance(v,np.ndarray)})
    separate=np.zeros((240,320,3),np.uint8); cv.circle(separate,(90,120),40,(255,255,255),3); cv.circle(separate,(235,120),30,(255,255,255),3)
    detected=lab06.task07_coins(separate,min_radius=20,max_radius=50,min_distance=60,param2=25)
    check('Synthetic circles detected',detected['count']==2); mark(6,7); save('lab06_circles',{'Hough circle detections':detected['overlay']},1)
    background=np.zeros((120,160,3),np.uint8); intruder=background.copy(); intruder[40:80,50:100]=255
    monitor=lab06.ZoneMonitor(background,[(20,20),(140,20),(140,100),(20,100)],persistence=2)
    one=monitor.update(intruder); two=monitor.update(intruder); three=monitor.update(intruder)
    check('Zone event waits for persistence and fires once',not one['new_event'] and two['new_event'] and not three['new_event'])
    check('Zone resets after object gone',not monitor.update(background)['active']); mark(6,8)
    # Corrected manual algorithms: independent reference for convolution.
    f=np.arange(25,dtype=np.float32).reshape(5,5); kernel=np.float32([[1,2,3],[0,0,0],[-1,-2,-3]])
    expected=np.array([[np.sum(f[y:y+3,x:x+3]*kernel[::-1,::-1]) for x in range(3)] for y in range(3)])
    check('Convolution agrees with direct mathematical sum',np.allclose(core.convolution(f,kernel,'valid'),expected))
    check('Full convolution output shape',core.convolution(f,kernel,'full').shape==(7,7))
    check('Stride applies after valid convolution',np.array_equal(core.convolution(f,kernel,'valid',2),expected[::2,::2]))
    filtering=manual_examples.filtering_examples(image)
    check('Separable and 2D Gaussian agree',np.allclose(filtering['gaussian2d'],filtering['gaussian_separable'],atol=1e-4))
    check('All manual function groups run',bool(manual_examples.basic_examples(image,image)) and bool(manual_examples.feature_examples(image)) and bool(manual_examples.segmentation_examples(coins)) and bool(manual_examples.edge_feature_examples(image)) and bool(manual_examples.morphology_examples(truth)))
    constant=np.full((64,64),128,np.uint8)
    check('Uniform image has no gradient histogram energy',np.allclose(core.direction_histogram(constant)[0],0))
    check('All 41 task implementations exercised',len(set(exercised))==41)
    report={'status':'passed','checks_passed':len(checks),'checks':checks,'tasks_exercised':sorted(exercised),
            'data':'Generated synthetic fixtures only; original course images/videos absent.',
            'versions':{'python':platform.python_version(),'opencv':cv.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            'document_iou':comparison['iou'],'best_document_method':comparison['best_on_supplied_truth'],
            'otsu_thresholds':otsu['thresholds'],
            'distance_sweep':{f:{k:r[k] for k in ['seed_count','region_count']} for f,r in dsweep.items()},
            'screen_results':sr['screens'],'synthetic_circle_count':detected['count'],
            'wavelet_flagged_indices':anomaly['anomaly_indices'].tolist(),
            'limits':['No real-data accuracy measured','Optional Pandas path not exercised','Only installed versions tested','No network API is used by knowledge search or solutions']}
    (ROOT/'reports'/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','checks':len(checks),'tasks':len(set(exercised)),'document_iou':comparison['iou']}))

if __name__=='__main__': main()
