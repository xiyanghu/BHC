from BHClustering.functions import bhc, draw_dendrogram

import numpy as np

d1 = np.random.dirichlet((20, 1), 2)
d2 = np.random.dirichlet((1, 100), 2)
d3 = np.random.dirichlet((100, 100), 2)
sdata = np.concatenate((d1,d2,d3),axis=0).reshape(-1,1)

r2,r3 = bhc(sdata, alpha = 500, r_thres = 0.5)

print(r3)

z = np.array(r3[-3], dtype=float)
plt.figure(tight_layout=True, facecolor='white')
plt.scatter(sdata[:, 0], 1- sdata[:, 0], c=z, cmap='Set1', s=225)
plt.show()
