import numpy as np
a = np.array([[1, 2], [3, 4]])
print("sum of array")
print(np.sum(a)) # sum of all array elements
print(np.sum(a, axis=0)) # sum of each column
print(np.sum(a, axis=1)) # sum of each row
print("\nmean of array")
print(np.mean(a)) # mean of all array elements
print(np.mean(a, axis=0)) # mean of each column
print(np.mean(a, axis=1))