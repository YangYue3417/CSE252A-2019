import numpy as np
from scipy.misc import imread
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import imageio


def rgb2gray(rgb):
    """ Convert rgb image to grayscale.
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def corner_detect(image, nCorners, smoothSTD, windowSize):
    """Detect corners on a given image.

    Args:
        image: Given a grayscale image on which to detect corners.
        nCorners: Total number of corners to be extracted.
        smoothSTD: Standard deviation of the Gaussian smoothing kernel.
        windowSize: Window size for corner detector and non maximum suppression.

    Returns:
        Detected corners (in image coordinate) in a numpy array (n*2).

    """

    """
    Put your awesome numpy powered code here:
    """
    radi = windowSize // 2
    img_smth = gaussian_filter(image, sigma=smoothSTD)

    dx_kernel = np.array([[-0.5, 0, 0.5]])
    dx_img = convolve(img_smth, dx_kernel, mode='same')
    dx_img[:, 0] = dx_img[:, 1]
    dx_img[:, -1] = dx_img[:, -2]

    dy_kernel = np.array([[-0.5, 0, 0.5]]).T
    dy_img = convolve(img_smth, dy_kernel, mode='same')
    dy_img[0, :] = dy_img[1, :]
    dy_img[-1, :] = dy_img[-2, :]

    C_lambda = np.zeros([image.shape[0], image.shape[1]])

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            top = 0 if (row - radi < 0) else row - radi
            bottom = image.shape[0] if (radi + row > image.shape[0]) else radi + row
            left = 0 if (col - radi) < 0 else col - radi
            right = image.shape[1] if (radi + col > image.shape[1]) else radi + col

            dxWindow = dx_img[top:bottom + 1, left:right + 1]
            dyWindow = dy_img[top:bottom + 1, left:right + 1]

            Ix = np.sum(dxWindow * dxWindow)
            Iy = np.sum(dyWindow * dyWindow)
            Ixy = np.sum(dxWindow * dyWindow)
            c = np.array([[Ix, Ixy], [Ixy, Iy]])
            C_lambda[row, col] = min(np.linalg.eigvals(c))

    # nms
    # C_nms = []
    C_nms = np.array([0, 0, 0])
    for row in range(0, image.shape[0], windowSize):
        for col in range(0, image.shape[1], windowSize):
            # for row in range(image.shape[0]):
            #     for col in range(image.shape[1]):
            top = 0 if (row - radi < 0) else row - radi
            bottom = image.shape[0] if (radi + row > image.shape[0]) else radi + row
            left = 0 if (col - radi) < 0 else col - radi
            right = image.shape[1] if (radi + col > image.shape[1]) else radi + col
            cWindow = C_lambda[top:bottom + 1, left:right + 1]

            maxLambda = max(cWindow.flatten())
            winRow = np.argmax(cWindow) // (bottom - top + 1)
            winCol = np.argmax(cWindow) - winRow * (bottom - top + 1)
            maxPosGlb = [top + winRow, left + winCol]
            # C_nms.append((maxLambda, maxPosGlb[1], maxPosGlb[0]))
            C_nms = np.vstack((C_nms, np.array([maxLambda, maxPosGlb[1], maxPosGlb[0]])))

    # C_nms = np.asarray(C_nms)
    C_nms = np.unique(C_nms, axis=0)
    # C_nms = np.flip(C_nms, 0)
    # C_nms.tolist()
    #     C_nms.sort(reverse=True)
    # data = data[data[:,2].argsort()]
    # C_nms = C_nms[-C_nms[:,0].argsort()]

    C_nms_sort = C_nms[np.lexsort(-C_nms[:, ::-1].T)]

    corners = np.zeros((nCorners, 2))
    for rowCorner in range(nCorners):
        corners[rowCorner][0] = C_nms_sort[rowCorner][1]
        corners[rowCorner][1] = C_nms_sort[rowCorner][2]

    return corners


def show_corners_result(imgs, corners):
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax1.imshow(imgs[0], cmap='gray')
    ax1.scatter(corners[0][:, 0], corners[0][:, 1], s=36, edgecolors='r', facecolors='none')

    ax2 = fig.add_subplot(222)
    ax2.imshow(imgs[1], cmap='gray')
    ax2.scatter(corners[1][:, 0], corners[1][:, 1], s=36, edgecolors='r', facecolors='none')
    plt.show()


# detect corners on warrior and matrix image sets
# adjust your corner detection parameters here
nCorners = 20
smoothSTDs = [0.5, 1, 2, 4]
imgs_mat = []
imgs_war = []
grayimgs_mat = []
grayimgs_war = []
# Read the two images and convert it to Greyscale
for i in range(2):
    img_mat = imageio.imread('p4/matrix/matrix' + str(i) + '.png')
    imgs_mat.append(img_mat)
    grayimgs_mat.append(rgb2gray(img_mat))
    # Comment above line and uncomment below line to
    # downsize your image in case corner_detect runs slow in test
    # grayimgs_mat.append(rgb2gray(img_mat)[::2, ::2])
    # if you unleash the power of numpy you wouldn't need to downsize, it'll be fast
    img_war = imageio.imread('p4/warrior/warrior' + str(i) + '.png')
    imgs_war.append(img_war)
    grayimgs_war.append(rgb2gray(img_war))

for smoothSTD in smoothSTDs:
    windowSize = int(6 * smoothSTD)
    if windowSize % 2 == 0: windowSize += 1
    crns_mat = []
    crns_war = []
    print("SmoothSTD:", smoothSTD, "WindowSize:", windowSize)
    for i in range(2):
        crns_mat.append(corner_detect(grayimgs_mat[i], nCorners, smoothSTD, \
                                      windowSize))
        # crns_war.append(corner_detect(grayimgs_war[i], nCorners, smoothSTD, \
        #                               windowSize))
    show_corners_result(imgs_mat, crns_mat)  # uncomment this to show your output!
    # show_corners_result(imgs_war, crns_war)


def ncc_match(img1, img2, c1, c2, R):
    """Compute NCC given two windows.

    Args:
        img1: Image 1.
        img2: Image 2.
        c1: Center (in image coordinate) of the window in image 1.
        c2: Center (in image coordinate) of the window in image 2.
        R: R is the radius of the patch, 2 * R + 1 is the window size

    Returns:
        NCC matching score for two input windows.

    """

    """
    Your code here:
    """
    matching_score = 0

    [w1_top, w1_left] = c1 - R
    [w1_bottom, w1_right] = c1 + R + 1
    [w2_top, w2_left] = c2 - R
    [w2_bottom, w2_right] = c2 + R + 1

    window1 = img1[w1_left:w1_right, w1_top:w1_bottom]
    window2 = img2[w2_left:w2_right, w2_top:w2_bottom]

    W1_mean = np.mean(window1)
    W2_mean = np.mean(window2)

    temp1 = np.sqrt(np.sum(np.square(window1 - W1_mean)))
    temp2 = np.sqrt(np.sum(np.square(window2 - W2_mean)))

    for row in range(window1.shape[0]):
        for col in range(window1.shape[1]):
            w1_temp = (window1[row, col] - W1_mean) / temp1
            w2_temp = (window2[row, col] - W2_mean) / temp2
            matching_score += w1_temp * w2_temp

    return matching_score
