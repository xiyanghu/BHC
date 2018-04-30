from BHClustering.functions import bhc, draw_dendrogram

import csv
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
dataset=dataset.drop(['class'], axis=1)
ar_iris = np.array(dataset,dtype=None)[:, 0:3]
df_new = np.array(ar_iris)
df_new_copy = np.array(ar_iris)
sum_df = np.sum(df_new, axis=1)
df_new2 =np.array([])
for i in range(df_new.shape[0]):
    df_new2 = np.append(df_new2,np.array(df_new[i,:]/sum_df[i]))
df_new2 = df_new2.reshape(-1,3)
df_new2 = np.concatenate((df_new2[0:10,:], df_new2[50:60,:], df_new2[100:110,:]))

r2,r3 = bhc(df_new2, alpha = 100, r_thres = 0.5)

draw_dendrogram(r2)