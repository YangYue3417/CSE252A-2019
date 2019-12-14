# this line prepares IPython for working with matplotlib
# import matplotlib
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
x = np.arange(-24, 24) / 24. * math.pi
plt.plot(x, np.sin(x))
plt.xlabel('radians')
plt.ylabel('sin value')
plt.title('Sine Function')
plt.show()
