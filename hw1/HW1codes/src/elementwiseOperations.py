import numpy as np
a = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float64)
print(a * 2) # scalar multiplication
print(a / 4) # scalar division
print(np.round(a / 4)) # 四舍五入
print(np.power(a, 2))
print(np.log(a))
b = np.array([[5, 6, 7], [5, 7, 8]], dtype=np.float64)
print(a + b) # elementwise sum
print(a - b) # elementwise difference
print(a * b) # elementwise product
print(a / b)