# coding: utf-8

import pandas as pd
from skimage import io
import glob
import os, sys
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog

# train_csv = pd.read_csv('input_datas/train_v2.csv')
# print(train_csv)

input_file_path = os.path.join(os.path.dirname(__file__), 'input_file/*.jpg')
filepaths = glob.glob(input_file_path)
print(filepaths)
imgs = [io.imread(f) for f in filepaths]
for img in imgs:
    img_cnl = img[:,:,0], img[:,:,1], img[:,:,2]
    img_hog = np.array([ hog(cnl, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)) for cnl in img_cnl])
    img_hog = img_hog.flatten()
