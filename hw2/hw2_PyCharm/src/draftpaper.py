import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

target_img = np.array(Image.open("myop.png"))/255
print(target_img)