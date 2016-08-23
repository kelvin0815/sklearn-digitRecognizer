# Import the modules
import timeit
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.svm import LinearSVC
from collections import Counter
from feature import HOG

###################################################
############      TRAIN SVM MODEL      ############
###################################################

# Load scikit-learn MNIST handwritting digit dataset
# Reference: http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
mnist = datasets.fetch_mldata("MNIST Original")

# Get pixel value and label of MNIST dataset 
pixels = np.array(mnist.data, 'uint8')
labels = np.array(mnist.target, 'int')

# Extract HOG feature vectors from training set
start = timeit.default_timer()
print "--HOG Extracton Ongoing..."
hog_fvs = []
for px in pixels:
    fd = HOG(px.reshape((28, 28)), numOfCells=(4,4), numOfBins=9, gamma=False)[0]
    hog_fvs.append(fd)
hog_features = np.array(hog_fvs, 'float64')
end = timeit.default_timer()
print "--HOG Extracton Ends"

print "Count of digits in dataset", Counter(labels)
print "HOG Extraction Time: " + str(round(end-start, 4)) + 's'

# Create Support Vector Machine (SVM) object
clf = LinearSVC()

# Training Model
start = timeit.default_timer()
print "--SVM Model Ongoing..."
clf.fit(hog_features, labels)
end = timeit.default_timer()
print "--SVM Model Ends"

print "Model Training Time: " + str(round(end-start, 4)) + 's'

# Save the classifier
joblib.dump(clf, "hog4.pkl", compress=3)