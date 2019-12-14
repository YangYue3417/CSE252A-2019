# Setup
import pickle
import numpy as np
from time import time
from skimage import io
#%matplotlib inline
import matplotlib.pyplot as plt

### Example: how to read and access data from a .pickle file
#pickle_in = open("synthetic_data.pickle", "rb")
#data = pickle.load(pickle_in, encoding="latin1")

#print("Keys: ", list(data.keys()))

pickle_in_pear = open("specular_pear.pickle","rb")
pear_data = pickle.load(pickle_in_pear,encoding="latin1")

lights_pear = np.vstack((pear_data["l1"], pear_data["l2"], pear_data["l3"], pear_data["l4"]))

images_s = []
images_s.append(pear_data["im1"])
images_s.append(pear_data["im2"])
images_s.append(pear_data["im3"])
images_s.append(pear_data["im4"])
images_s = np.array(images_s)

mask_s = np.ones(pear_data["im1"].shape)

pickle_in = open("synthetic_data.pickle", "rb")
data = pickle.load(pickle_in, encoding="latin1")

lights = np.vstack((data["l1"], data["l2"], data["l4"]))
# lights = np.vstack((data["l1"], data["l2"], data["l3"], data["l4"]))

images = []
images.append(data["im1"])
images.append(data["im2"])
# images.append(data["im3"])
images.append(data["im4"])
images = np.array(images)

mask = np.ones(data["im1"].shape)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

im1_s_gray = rgb2gray(pear_data["im1"])
im2_s_gray = rgb2gray(pear_data["im2"])
im3_s_gray = rgb2gray(pear_data["im3"])
im4_s_gray = rgb2gray(pear_data["im4"])

print("The original, S channel and Diffuse part of Sphere:")