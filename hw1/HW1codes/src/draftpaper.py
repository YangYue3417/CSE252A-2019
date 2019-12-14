import numpy as np

a = np.array([[1,3,2],[4,2,3],[1,2,4]])
t = np.array([[1],[1],[1]])
a = np.hstack((a, t))

print(a[:,:])