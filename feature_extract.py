# coding: utf-8

import numpy as np
from skimage.feature import hog

'''
    Feature extraction from np.array
'''
def hog_cnl(image, channels=3):
    img_cnl = [ image[:, :, i] for i in range(channels)]
    img_hog = np.array([ hog(cnl, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)) for cnl in img_cnl])
    return img_hog.flatten()

def bovw(image):
    pass
