"""Lab 01 solutions. Input images must be supplied; see TASK_SOLUTIONS.md."""
import cv2 as cv
import numpy as np
from cv_core import gray, grid, read_image, warp_fit


class GroceryManager:
    """Task 1. Re-adding an item adds quantity and updates its unit price."""
    def __init__(self): self.items = {}
    def add_item(self, item, quantity, price):
        if not item or quantity <= 0 or price < 0: raise ValueError('Invalid item, quantity or price')
        previous = self.items.get(item, {'quantity':0})['quantity']
        self.items[item] = {'quantity':previous+quantity, 'price':float(price)}
    def remove_item(self,item):
        if item not in self.items: raise KeyError(f'Item not present: {item}')
        return self.items.pop(item)
    def view_list(self): return {name:info.copy() for name,info in self.items.items()}
    def calculate_total(self): return sum(x['quantity']*x['price'] for x in self.items.values())


def task02_students(students, major):
    """Task 2. Name/Major/Grades keys; Grades a list or subject->grade mapping.
    Tied highest averages return all names; students with no grades are excluded.
    """
    means = {}
    for sid, info in students.items():
        grades = info['Grades']
        if isinstance(grades,dict): grades=list(grades.values())
        if grades: means[sid]=float(np.mean(grades))
    best=max(means.values(),default=None)
    return {'highest_average':best,
            'top_names':[students[s]['Name'] for s,m in means.items() if m==best],
            'major_matches':{s:i for s,i in students.items() if i['Major'].casefold()==major.casefold()}}


def task03_display(path, output='task03.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    im=read_image(path)
    fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
    ax.set_title('King of Curses — Ryomen Sukuna',fontsize=16,color='darkred',pad=15)
    ax.axis('off'); fig.savefig(output,bbox_inches='tight'); plt.close(fig)
    return im


def task04_circles():
    im=np.zeros((800,800,3),np.uint8); center=(400,400)
    for i,radius in enumerate([300,240,180,120,60]):
        cv.circle(im,center,radius,(255,255,255) if i%2==0 else (0,0,0),-1)
    cv.rectangle(im,(100,100),(700,700),(0,255,0),2)
    return {'image':im,'drawing_center':center,'geometric_pixel_center':(399.5,399.5),
            'bounding_box_xyxy':(100,100,700,700)}


def task05_blur_roi(image):
    h,w=image.shape[:2]
    if min(h,w)<300: raise ValueError('Image must be at least 300x300')
    blurred=cv.GaussianBlur(image,(25,25),0)
    x,y=(w-300)//2,(h-300)//2
    return {'Original center 300x300':image[y:y+300,x:x+300],
            'Blurred center 300x300':blurred[y:y+300,x:x+300]}


def task06_overlay(image, opacity=0.4, text='Computer Vision Lab'):
    if not 0<=opacity<=1: raise ValueError('opacity must be in [0,1]')
    overlay=image.copy(); h,w=image.shape[:2]; y=int(h*0.8)
    overlay[y:]=(255,0,0)
    out=cv.addWeighted(overlay,opacity,image,1-opacity,0)
    cv.putText(out,text,(10,min(h-5,y+max(15,(h-y)//2))),cv.FONT_HERSHEY_SIMPLEX,
               max(0.3,w/1000),(255,255,255),1,cv.LINE_AA)
    return out


def task07_threshold_rotate(image):
    g=gray(image); h,w=g.shape
    _,fixed=cv.threshold(g,127,255,cv.THRESH_BINARY)
    adaptive=cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,7)
    M=cv.getRotationMatrix2D(((w-1)/2,(h-1)/2),45,0.8)
    H=np.vstack([M,[0,0,1]])
    rotated,_=warp_fit(image,H)
    return {'Grayscale':g,'Global 127':fixed,'Adaptive':adaptive,'45deg scale0.8 expanded':rotated}


def task08_composite(foreground,background):
    a=cv.resize(foreground,(500,500)); b=cv.resize(background,(500,500))
    mask=np.zeros((500,500),np.uint8); cv.circle(mask,(250,250),180,255,-1)
    fg=cv.bitwise_and(a,a,mask=mask); bg=cv.bitwise_and(b,b,mask=cv.bitwise_not(mask))
    return {'Mask':mask,'Foreground':fg,'Background':bg,'Composite':cv.bitwise_or(fg,bg)}


def task09_rgb_statistics(image):
    pixels=cv.cvtColor(image,cv.COLOR_BGR2RGB).reshape(-1,3).astype(float)
    result={}
    for i,name in enumerate(('R','G','B')):
        v=pixels[:,i]; q=np.quantile(v,[0,.25,.5,.75,1])
        result[name]=dict(zip(['count','mean','std','min','25%','50%','75%','max'],
                              [len(v),float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else None,*map(float,q)]))
    return result


def task09_optional_pandas(image):
    """Original task explicitly asks for Pandas; optional dependency, outside core."""
    import pandas as pd
    rgb=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    return pd.DataFrame(rgb.reshape(-1,3),columns=['R','G','B']).describe()
