"""Embedded material-texture task; handcrafted features need labeled examples."""
import numpy as np
from cv_core import uniform_lbp,hog_features,texture_stats


def task01_features(image):
    labels,lbp=uniform_lbp(image); energy,contrast=texture_stats(image)
    return {'lbp_image':labels,'lbp_histogram':lbp,'hog':hog_features(image),
            'energy':energy,'local_contrast':contrast}


def classify_material(image,labeled_images):
    """labeled_images=[('wood',image),...]. Nearest histogram baseline, not a trained general model."""
    if not labeled_images: raise ValueError('Provide labeled training examples')
    query=uniform_lbp(image)[1]; scores=[]
    for label,sample in labeled_images:
        h=uniform_lbp(sample)[1]
        distance=0.5*float(np.sum((query-h)**2/(query+h+1e-8)))
        scores.append((distance,label))
    scores.sort()
    return {'label':scores[0][1],'distance':scores[0][0], 'all_scores':scores}
