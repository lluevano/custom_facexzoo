import numpy as np

def compute_AP(good_image, index):
    cmc = np.zeros((len(index),))
    ngood = len(good_image)
    
    old_recall = 0
    old_precision = 1.0
    ap = 0
    good_now = 0
    intersect_size = good_now
    
    
    for n in range(len(index)):
        if index[n] in good_image:
            cmc[n:] = 1
            good_now += 1
            intersect_size = good_now
        
        recall = intersect_size/float(ngood)
        precision = intersect_size/float(n+1)
        ap = ap + (recall - old_recall)*((old_precision + precision)/2.0)
        old_recall = recall
        old_precision = precision
        
        if good_now == ngood:
            return ap, cmc
