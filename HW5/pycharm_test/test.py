import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm_notebook


def grayscale(img):
    '''
    Converts RGB image to Grayscale
    '''
    gray = np.zeros((img.shape[0], img.shape[1]))
    gray = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140
    return gray


def plot_optical_flow(img, U, V, titleStr):
    '''
    Plots optical flow given U,V and one of the images
    '''

    # Change t if required, affects the number of arrows
    # t should be between 1 and min(U.shape[0],U.shape[1])
    t = 10

    # Subsample U and V to get visually pleasing output
    U1 = U[::t, ::t]
    V1 = V[::t, ::t]

    # Create meshgrid of subsampled coordinates
    r, c = img.shape[0], img.shape[1]
    cols, rows = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    cols = cols[::t, ::t]
    rows = rows[::t, ::t]

    # Plot optical flow
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.quiver(cols, rows, U1, V1)
    plt.title(titleStr)
    plt.show()


images = []
for i in range(1, 5):
    images.append(plt.imread('optical_flow_images/im' + str(i) + '.png')[:, :288, :])


# each image after converting to gray scale is of size -> 400x288

# you can use interpolate from scipy
# You can implement 'upsample_flow' and 'OpticalFlowRefine'
# as 2 building blocks in order to complete this.
def upsample_flow(u_prev, v_prev):
    ''' You may implement this method to upsample optical flow from
    previous level
    u_prev, v_prev -> optical flow from prev level
    u, v -> upsampled optical flow to the current level
    '''
    """ ==========
    YOUR CODE HERE
    ========== """
    x = np.arange(u_prev.shape[0]) * 2
    y = np.arange(v_prev.shape[1]) * 2
    u_interp = interpolate.interp2d(x, y, u_prev.T, kind='linear')
    v_interp = interpolate.interp2d(x, y, v_prev.T, kind='linear')
    x_new = np.arange(u_prev.shape[0] * 2)
    y_new = np.arange(u_prev.shape[1] * 2)
    u = u_interp(x_new, y_new).T
    v = v_interp(x_new, y_new).T

    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    print("u.shape: ", u.shape)
    print("v.shape: ", v.shape)
    return u, v


def OpticalFlowRefine(im1, im2, window, u_prev=None, v_prev=None):
    '''
    Inputs: the two images at current level and window size
    u_prev, v_prev - previous levels optical flow
    Return u,v - optical flow at current level
    '''
    # upsample flow from previous level
    u_prev, v_prev = upsample_flow(u_prev, v_prev)
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    """ ==========
    YOUR CODE HERE
    ========== """
    print("im2.shape: ", im2.shape)

    Iy, Ix = np.gradient(im1)

    ixx = Ix * Ix
    iyy = Iy * Iy
    ixy = Ix * Iy

    radi = window // 2

    for row in range(im1.shape[0]):
        for col in range(im1.shape[1]):
            top = 0 if (row - radi < 0) else row - radi
            bottom = im1.shape[0] if (radi + row > im1.shape[0]) else radi + row
            left = 0 if (col - radi) < 0 else col - radi
            right = im1.shape[1] if (radi + col > im1.shape[1]) else radi + col

            Ix2 = np.sum(ixx[top:bottom + 1, left:right + 1])
            Ixy = np.sum(ixy[top:bottom + 1, left:right + 1])
            Iy2 = np.sum(iyy[top:bottom + 1, left:right + 1])

            #             print("top: ",top," bottom: ",bottom, "left: ",left," right: ", right)
            #             print("u_prev[row,col]: ", int(u_prev[row,col]),"v_prev[row,col]: ",int(v_prev[row,col]))
            im2_window = im2[top + int(v_prev[row, col]):bottom + 1 + int(v_prev[row, col]), \
                         left + int(u_prev[row, col]):right + 1 + int(u_prev[row, col])]
            It = im2_window - im1[top:bottom + 1, left:right + 1]
            ixt = Ix[top:bottom + 1, left:right + 1] * It
            iyt = Iy[top:bottom + 1, left:right + 1] * It
            Ixt = np.sum(ixt)
            Iyt = np.sum(iyt)

            M = np.array([[Ix2, Ixy], [Ixy, Iy2]])
            b = np.array([[-Ixt, -Iyt]])
            uv_vec = np.dot(np.linalg.pinv(M), b.T)

            u[row, col] = uv_vec[0]
            v[row, col] = uv_vec[1]

    print(u.shape)
    u = u + u_prev
    v = v + v_prev
    return u, v


def LucasKanadeMultiScale(im1, im2, window, numLevels=2):
    '''
    Implement the multi-resolution Lucas kanade algorithm
    Inputs: the two images, window size and number of levels
    if numLevels = 1, then compute optical flow at only the given image level.
    Returns: u, v - the optical flow
    '''

    """ ==========
    YOUR CODE HERE
    ========== """
    # You can call OpticalFlowRefine iteratively
    layer = numLevels
    init_temp = 2 ** layer
    u_prev = np.floor(np.zeros([int(im1.shape[0] / init_temp), int(im1.shape[1] / init_temp)]))
    v_prev = np.floor(np.zeros_like(u_prev))

    for i in range(numLevels):
        layer -= 1
        im1_resize = im1[::2 ** layer, ::2 ** layer]
        im2_resize = im2[::2 ** layer, ::2 ** layer]

        u, v = OpticalFlowRefine(im1_resize, im2_resize, window, u_prev, v_prev)
        u_prev = u
        v_prev = v

    return u, v


window = 13
numLevels = 3
# Plot
U, V = LucasKanadeMultiScale(grayscale(images[0]), grayscale(images[1]), window, numLevels)
plot_optical_flow(images[0], U, V, 'levels = ' + str(numLevels) + ', window = ' + str(window))
