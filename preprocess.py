# Import modules
import cv2
import numpy as np

#################################################################
##############     IMAGE PREPROCESSING FUNCTION    ##############
#################################################################

def boxFilter(img, filterSize):
    """
    Box Filtering

    Args:
    1. img, ndarray, the input image
    2. filterSize, int, the size of filter

    Return:
    1. output_img, ndarray, the output image after blurring
    """
    output_img = img.copy()
    nRow, nCol = img.shape
    c = 1.0/filterSize**2
    a = filterSize/2
    bft = c * np.array([1 for _ in xrange(filterSize**2)]).astype('float32')
    for i in xrange(a, nRow-a):
        for j in xrange(a, nCol-a):
            output_img[i, j] = np.uint8(bft.reshape(int(1.0/c)).dot(img[i-a:i+a+1, j-a:j+a+1].reshape(int(1.0/c))))
    return output_img


def thresholding(img, threshold, flip):
    """
    Thresholding

    Args:
    1. img, ndarray, the input image
    2. threshold, int, range: 0-255
    3. flip, bool, set True if you want to perform image negative after thresholding

    Return:
    1. output_img, ndarray, the output image after thresholding
    """
    output_img = img.copy()
    nRow, nCol = img.shape
    if flip:
        thresMap = [255.0 if i < threshold else 0 for i in xrange(256)]
    else:
        thresMap = [255.0 if i > threshold else 0 for i in xrange(256)]
    for i in xrange(nRow):
        for j in xrange(nCol):
            output_img[i, j] = np.uint8(thresMap[int(img[i, j])])
    return output_img


# Sobel Transform
def sobel3(img):
    """
    Perform Sobel tranform using 3 x 3 kernel

    Args:
    1. img, ndarray, the input image

    Return:
    1. Gx, ndarray, the gradient in x direction
    2. Gy, ndarray, the gradient in y direction
    """
    nRow, nCol = img.shape
    Gx = np.zeros((nRow, nCol))
    Gy = np.zeros((nRow, nCol))
    xFilter = np.array([-1., 0., 1., -2., 0., 2., -1., 0., 1.]).astype('float32')
    yFilter = np.array([-1., -2., -1., 0., 0., 0., 1., 2., 1.]).astype('float32')
    
    for i in xrange(1, nRow-1):
        for j in xrange(1, nCol-1):
            Gx[i, j] = xFilter.reshape(9).dot(img[i-1:i+2, j-1:j+2].reshape(9))
    for i in xrange(1, nRow-1):
        for j in xrange(1, nCol-1):
            Gy[i, j] = yFilter.reshape(9).dot(img[i-1:i+2, j-1:j+2].reshape(9))
    
    return Gx, Gy