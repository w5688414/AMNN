from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import pandas as pd
import cv2
import os
from tqdm import tqdm
import h5py

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()

print("[INFO] describing images...")
data_path='./data/HARRISON/'
training_filename = data_path + 'training_data.txt'
image_path='/home/eric/data/HARRISON'
print('Loading training dataset...')
train_data = pd.read_table(training_filename, delimiter='*')
# train_data.drop(columns=['tweets'],inplace=True)
print(train_data.iloc[0,1])
print(train_data.head())
train_data = train_data.values.tolist()
# train_data = np.array(train_data,dtype=str) //problem on transform numpy
print(train_data[0][1])
# imagePaths = list(paths.list_images(args["dataset"]))
print(len(train_data))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

for img_file,hashatgs in tqdm(train_data):

    img_file_path=os.path.join(image_path,img_file)
    image = cv2.imread(img_file_path)
    pixels = image_to_feature_vector(image)
    # print(img_file)
    # print(hashatgs)
    list_hashtag=hashatgs.strip().split()
    rawImages.append(pixels)
    labels.append(list_hashtag)

f_out = h5py.File(os.path.join(data_path,"train.h5"), "w")
for i in range(len(train_data)):
    f_out.create_dataset(name=train_data[i][0], data=rawImages[i])