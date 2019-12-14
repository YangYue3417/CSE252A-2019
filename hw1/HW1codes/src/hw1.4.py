import matplotlib.pyplot as plt
import copy
import copy

import matplotlib.pyplot as plt

img = plt.imread('Lenna.png') # read a JPEG image
print("Image shape", img.shape) # print image size and color depth
plt.imshow(img) # displaying the original image
plt.show()

def iterative(img):
    image = copy.deepcopy(img) # create a copy of the image matrix
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if x < image.shape[0]/2 and y < image.shape[1]/2:
                image[x,y] = image[x,y] * [0,1,1] #removing the red channel
            elif x > image.shape[0]/2 and y < image.shape[1]/2:
                image[x,y] = image[x,y] * [1,0,1] #removing the green channel
            elif x < image.shape[0]/2 and y > image.shape[1]/2:
                image[x,y] = image[x,y] * [1,1,0] #removing the blue channel
            else:
                pass
    return image

def vectorized(img):
image = copy.deepcopy(img)
a = int(image.shape[0]/2)
b = int(image.shape[1]/2)
image[:a,:b] = image[:a,:b]*[0,1,1]
image[a:,:b] = image[a:,:b]*[1,0,1]
image[:a,b:] = image[:a,b:]*[1,1,0]
return image