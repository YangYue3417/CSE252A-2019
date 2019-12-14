import numpy as np

array1d = np.array([1, 0, 0])  # a 1d array
print("1d array :")
print(array1d)
print("Shape :", array1d.shape)  # print the shape of array

array2d = np.array([[1], [2], [3]])  # a 2d array
print("\n2d array :")
print(array2d)
print("Shape :", array2d.shape)  # print the size of v, notice the difference
print("\nTranspose 2d :", array2d.T)  # Transpose of a 2d array
print("Shape :", array2d.T.shape)
print("\nTranspose 1d :", array1d.T)  # Notice how 1d array did not change, after transpose (Thoughts?)
print("Shape :", array1d.T.shape)
allzeros = np.zeros([2, 3])  # a 2x3 array of zeros
allones = np.ones([1, 3])  # a 1x3 array of ones
identity = np.eye(3)  # identity matrix
rand3_1 = np.random.rand(3, 1)  # random matrix with values in [0, 1]
arr = np.ones(allones.shape) * 3  # create a matrix from shape
