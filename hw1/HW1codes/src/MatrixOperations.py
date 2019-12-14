import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("matrix-matrix product")
print(a.dot(b)) # matrix product
print(a.T.dot(b.T))
x = np.array([1, 2])

print("\nmatrix-vector product")
print(a.dot(x)) # matrix / vector product
print(a@x) # Can also make use of the @ instad of .dot()