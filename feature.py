# Import modules
import cv2
import numpy as np
from math import floor
from preprocess import sobel3


##########################################################
########      Histogram of Oriented Gradient      ########
##########################################################

def HOG(img, numOfCells=(4,4), numOfBins=9, gamma=False):
    """
    Histogram of Oriented Gradients

    Args: 
    1. img, ndarray, the input image
    2. numOfCells, tuple of int, the number of cells in x and y direction
    3. numOfBins, int, the number of bins of orientation
    4. gamma, bool, set True if perform gamma compression (gamma=0.5)

    Return:
    1. HOG, ndarray, a numOfCells[0]*numOfCells[1]*numOfBins HOG feature vector of img
    """

    # Size of cell
    nCellx = img.shape[0] / numOfCells[0]
    nCelly = img.shape[1] / numOfCells[1]

    # Size of bin
    binSize = (2 * np.pi) / numOfBins

    # Normalization Constant
    NORMAL_CONST = 1.0/(img.shape[0] * img.shape[1])

    # Initialize HOG feature vector
    n_HOG = numOfCells[0] * numOfCells[1] * numOfBins
    hog_fv = np.zeros((n_HOG, 1))

    # Stage 1: Gamma Compression (gamma=0.5)
    if gamma:
        img = np.sqrt(img)
    
    # Stage 2: Compute Image Gradient
    Gx, Gy = sobel3(img)
    orient = np.arctan2(Gy, Gx)
    mag = np.sqrt(Gx**2+Gy**2)

    # Stage 3: Compute Gradient Histogram
    # Stage 4: Normalise Gradient Value
    # Stage 5: Flatten into 1D feature vector
    for p in xrange(numOfCells[0]):
        for q in xrange(numOfCells[1]):
            for i in xrange(nCellx):
                for j in xrange(nCelly):

                    tmpX = p * nCellx + i
                    tmpY = q * nCelly + j

                    grad_mag = mag[tmpX, tmpY]

                    # Normalized Gradient Magitude
                    norm_grad = grad_mag * NORMAL_CONST

                    # Orientation
                    degree = orient[tmpX, tmpY]

                    # Mapping from (-pi,pi) to (0, 2*pi)
                    if degree < 0:
                        degree += 2 * np.pi

                    nth_bin = floor(degree / binSize)
                    hog_fv[((p * numOfCells[0] + q) * numOfBins + int(nth_bin))] += norm_grad

    return hog_fv.transpose()