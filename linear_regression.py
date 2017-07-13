# coding: utf-8

import pandas as pd
import numpy as np
import data_load
import feature_extract

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn

if __name__=="__main__":
    train_df = data_load.load_train()
    train_df = train_df.sample(40)
    X = [ data_load.load_image(row.image_name + ".jpg") for i, row in train_df.iterrows()]
    hog_X = np.array([ feature_extract.hog_cnl(x) for x in X ])
    y = np.array([ row.tags for i, row in train_df.iterrows()])
    print(hog_X.shape)
    print(y.shape)

    X_dev, X_val, y_dev, y_val = train_test_split(hog_X, y, test_size=0.2)

    pipe = Pipeline([('sc', StandardScaler()),
            ('pca', KernelPCA(n_components=200, kernel="linear")),
            ('clf', LinearRegression())])
