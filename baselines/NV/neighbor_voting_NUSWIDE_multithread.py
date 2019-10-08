# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import h5py
import pandas as pd
from tqdm import tqdm
import collections
import os
from threading import Thread
from time import time, sleep
from queue import Queue


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0])
    
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()

# load the images -- the original, the original + contrast,
# and the original + photoshop
root_path='/home/eric/data/102flowers/sorted/train'
img1 = cv2.imread(os.path.join(root_path,"0/image_06734.jpg"))
img2 = cv2.imread(os.path.join(root_path,'0/image_06735.jpg'))

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

img_feature1=image_to_feature_vector(img1)
img_feature2=image_to_feature_vector(img2)

data_path='./data/NUS-WIDE/'
image_features_filename=os.path.join(data_path,'inception_image_name_to_features.h5')
image_names_to_features = h5py.File(image_features_filename, 'r')
# print(image_names_to_features['dataset/fun/2018-12-28_09-41-29_UTC.jpg']['image_features'][:])

# data_path='/home/eric/Documents/Hashtag-recommendation-for-social-images/image_text_hashtagging/datasets/image_text/preprocessed_data/'
validation_filename = data_path + 'validation_data.txt'
image_path='/home/eric/data/NUS-WIDE/image'

log_filename=data_path+'predicted_hashtags.txt'

def write_to_file(filename,words, file='predicted_hashtags.txt'):
	f = open(file, 'a')
	f.write(filename+"*")
	list_hashtag=[]
	for item in words:
		list_hashtag.append(item[0])
		# f.write(str(item[0])+' ')
	f.write(' '.join(list_hashtag))
	f.write('\n')


def calculate(img_file):
	print(img_file)
	test_image_features = image_names_to_features[img_file]['image_features'][:]
	distance_dict={}
	for train_img, train_hashtag in tqdm(train_data):
		train_feature=image_names_to_features[train_img]['image_features'][:]
		result=mse(test_image_features,train_feature)
		distance_dict[train_img]=result
        # print(result)
    
	list_result=sorted(distance_dict.items(),key=lambda item:item[1])
	# print(list_result[:5])
	list_hashtags=[]
	for key,value in list_result[:5]:
        # print(train_dict[key])
		list_hashtags.extend(train_dict[key].strip().split())
	count_hashtag=collections.Counter(list_hashtags)
	word_freq=count_hashtag.most_common(5)
	write_to_file(img_file,word_freq,os.path.join(data_path,'predicted_hashtags.txt'))


class DownloadWorker(Thread):

    def __init__(self, queue,sleep=1):
        Thread.__init__(self)
        self.queue = queue
        self.numPicrures=0
        self.sleep = sleep

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            item = self.queue.get()
            if item is None:
                break
            # print(imageUrl)
            calculate(item)
            self.queue.task_done()
            # sleep(self.sleep)

training_filename = data_path + 'training_data.txt'
print('Loading training dataset...')
train_data = pd.read_table(training_filename, delimiter='*')
# train_data.drop(columns=['tweets'],inplace=True)
train_data = train_data.values.tolist()
print(len(train_data))

train_dict={}
for img_file,hashatgs in tqdm(train_data):
    train_dict[img_file]=hashatgs

if __name__ == "__main__":
	validation_data = pd.read_csv(validation_filename, delimiter='*')
	# validation_data.drop(columns=['tweets'],inplace=True)
	validation_data = validation_data.values.tolist()
	print(len(validation_data))


	log_data=pd.read_csv(log_filename, delimiter='*')
	log_data=log_data.iloc[:,0].values.tolist()
	
	queue = Queue()
	ts = time()
	
	seconds=0
	num_worker=1
	for x in range(num_worker):
		worker = DownloadWorker(queue,seconds)
        # Setting daemon to True will let the main thread exit even though the
        # workers are blocking
		worker.daemon = True
		worker.start()

	
	for img_file,hashtags in validation_data:
		if(img_file not in log_data):
			queue.put(img_file)
	print(queue.qsize())
	queue.join()

	
	print('Took {}s'.format(time() - ts))