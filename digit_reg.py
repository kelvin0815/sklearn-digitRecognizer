# Import modules
import sys
import timeit
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from feature import HOG
from preprocess import boxFilter, thresholding

######################################################
##############     DIGIT RECOGNITION    ##############
######################################################

# Load pre-trained SVM model
clf = joblib.load("hog4.pkl")

# Read grayscale image
img_path = sys.argv[1]
im_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print 'Input Image Size: ', im.shape

# Show original image
plt.imshow(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB))
plt.show()

# Box Filtering
fSize = 3

start = timeit.default_timer()
bft_img = boxFilter(im, fSize)
end = timeit.default_timer()
print 'Box Filtering Time: ' + str(round(end-start, 4)) + 's'

# Thresholding
thres = 90

start = timeit.default_timer()
thres_img = thresholding(bft_img, thres, True)
end = timeit.default_timer()
print 'Thresholding Time: ' + str(round(end-start, 4)) + 's'

# ROI & HOG
start = timeit.default_timer()
ctrs, hier = cv2.findContours(thres_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
rois = []
for rect in rects:
    maxSize = int(1.2*max(rect[2], rect[3]))
    # Center of rectangle
    Cx, Cy = (rect[1]+rect[3]/2, rect[0]+rect[2]/2)
    roi = thres_img[(Cx-maxSize/2):(Cx+maxSize/2), (Cy-maxSize/2):(Cy+maxSize/2)]
    
    if max(roi.shape) > 40:
        # Resize ROI to fit MNIST handwriting dataset
        roi = cv2.resize(roi, (28, 28))
        # Image Dilation
        roi = cv2.dilate(roi, (3, 3))
        # Draw ROI
        cv2.rectangle(im_color, (Cy-maxSize/2, Cx-maxSize/2), (Cy+maxSize/2, Cx+maxSize/2), (255, 0, 0), 3)
        # Compute HOG feature vector
        hog_feature = HOG(roi, numOfCells=(4,4), numOfBins=9, gamma=False)[0]
        predicted_digit = clf.predict(np.array([hog_feature], 'float64'))
        cv2.putText(im_color, str(predicted_digit[0]), (Cy+maxSize/2, Cx+maxSize/2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 255), 2)
        rois.append(roi)
end = timeit.default_timer()
print 'HOG Extraction Time: ' + str(round(end-start, 4)) + 's'

# Show image
plt.imshow(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB))
plt.show()