import numpy as np
array2d = np.array([[1, 2, 3], [4, 5, 6]]) # create a 2d array with shape (2, 3)
print("Access a single element")
print(array2d[0, 2]) # access an element
array2d[0, 2] = 252 # a slice of an array is a view into the same data;
print("\nModified a single element")
print(array2d) # this will modify the original array
print("\nAccess a subarray")
print(array2d[1, :]) # access a row (to 1d array)
print(array2d[1:, :]) # access a row (to 2d array)
print("\nTranspose a subarray")
print(array2d[1, :].T) # notice the difference of the dimension of resulting array
print(array2d[1:, :].T) # this will be helpful if you want to transpose it later
# Boolean array indexing
# Given a array m, create a new array with values equal to m
# if they are greater than 0, and equal to 0 if they less than or equal 0
array2d = np.array([[3, 5, -2], [50, -1, 0]])
arr = np.zeros(array2d.shape)
arr[array2d > 0] = array2d[array2d > 0]
print("\nBoolean array indexing")
print(arr)