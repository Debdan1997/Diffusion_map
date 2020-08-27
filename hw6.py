# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# Load data
a = pd.read_excel("HW_TESLA.xlt")

# Using all data points
x = np.array(a.iloc[:, 1:])
y = np.array(a.STATIC)

s = 1
pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
#pairwise_sq_dists = squareform(pdist(xTrain, 'sqeuclidean'))
K = np.exp(-pairwise_sq_dists / s) 

rowSum = np.sum(K,axis = 1)
D = np.diag(rowSum)

D_inv = np.linalg.inv(D)
D12 = np.sqrt(D)
D_12 = np.sqrt(D_inv)

P =  D_inv.dot(K)

q=10
for j in range(q):
    P=P.dot(P)

P_prime = (D12.dot(P)).dot(D_12)

# Check symmetric: 
# np.all(np.abs(P_prime-P_prime.T) < 1e-12))

eig_vals, eig_vecs = LA.eig(P_prime) # cols are eigen-vecs


#print(np.sum(eig_vals))
#print(np.sum(eig_vecs))

# extract real part of eig_vals, 
# Why? for s=100, its returning eig_vals containing complex part even though P_prime is
#   symmetric,
# JUST FOR TRIAL
eig_vals= eig_vals.real
eig_vecs = eig_vecs.real

# sort the eigen-vals and corresponding eigen-vecs
idx = eig_vals.argsort()[::-1]   
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:,idx] 

Eig_mat = np.diag(eig_vals)

plt.plot((np.cumsum(eig_vals)/np.sum(eig_vals))[:30])
plt.show()

print(np.sum(eig_vals)) 
# for s = 5
# around 77.22
# first 3 - 2.98

# for s = 100
# around 17.30
# first 100 - 16.96
# first 3 - 2.80
print(np.sum(eig_vals[:3]))

# obtain the data points in eigen basis of diffusion space
ek = D12.dot(eig_vecs)

# time for diffusion process
t = 1

# obtain data-points in eigen basis
v = D_12.dot(eig_vecs).dot(np.diag(np.power(eig_vals, t))) # each col is a datapoint

numComponents = 3
xDiffMapped = (v.T)[:, :numComponents]



#matplotlib.interactive(True)
#get_ipython().run_line_magic('matplotlib', 'qt')
fig = plt.figure()
ax = plt.axes(projection='3d')
#plt.xlim(-0.00037308308598523637, 2.346640416275214e-05)
ax.view_init(30, 60)
scatter = ax.scatter3D(xDiffMapped[:,0], xDiffMapped[:,1], xDiffMapped[:,2], c=y , label=y)
#ax.legend()

yZeroIdx = (y == 0)
xDiffMappedZero = xDiffMapped[yZeroIdx, :]

yOneIdx = (y == 1)
xDiffMappedOne = xDiffMapped[yOneIdx, :]

# FROM BELOW CAN  CONCLUDE THAT ALL 1's ARE BEING MAPPED IN A VERY SMALL 
# REGION, WHILE 0's ARE SCATTERED AROUND IT

# Print max-min value of datapoints belonging to 0 in y's in x-direction
# vertical is x-direction in plot, according to my observatoin
print(np.max(xDiffMappedZero[:, 0]))
print(np.min(xDiffMappedZero[:, 0]))

# Print max-min value of datapoints belonging to 1 in y's in x-direction
print(np.max(xDiffMappedOne[:, 0]))
print(np.min(xDiffMappedOne[:, 0]))
plt.show()