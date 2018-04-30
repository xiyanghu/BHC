from BHClustering.functions import bhc, draw_dendrogram

import csv
import numpy as np
import pandas as pd

filename = '../data/toyexample.csv'
import pandas as pd
import numpy as np
df=pd.read_csv(filename,header=None)

df_new = np.array(df)
df_new_copy = np.array(df)
sum_df = np.sum(df_new, axis=1)
df_new2 =np.array([])
for i in range(df_new.shape[0]):
    df_new2 = np.append(df_new2,np.array(df_new[i,:]/sum_df[i]))
df_new2 = df_new2.reshape(120,64)
df_new2 = np.concatenate((df_new2[0:5,:], df_new2[40:45,:], df_new2[80:85,:]))

r2,r3= bhc(df_new2, alpha = 500, r_thres = 0.5)

print(r3)

draw_dendrogram(r2)