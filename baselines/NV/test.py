import multiprocessing
from multiprocessing import Process, Queue
from tqdm import tqdm
import h5py
import os
from time import time, sleep
import pandas as pd
import collections
import numpy as np

print("Number of cpu : ", multiprocessing.cpu_count())
num_processes= multiprocessing.cpu_count()

data_path='./data/HARRISON/'
image_features_filename=os.path.join(data_path,'inception_image_name_to_features.h5')


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0])
    
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

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
    image_names_to_features = h5py.File(image_features_filename, 'r')
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

training_filename = data_path + 'training_data.txt'
print('Loading training dataset...')
train_data = pd.read_table(training_filename, delimiter='*')
# train_data.drop(columns=['tweets'],inplace=True)
train_data = train_data.values.tolist()
print(len(train_data))

train_dict={}
for img_file,hashatgs in tqdm(train_data):
    train_dict[img_file]=hashatgs

def run_main(proc_num, queue):
    print(proc_num, "Start loop")
    while True:
        if queue.empty():
            break
        item = queue.get()
        calculate(item)


validation_filename = data_path + 'validation_data.txt'
validation_data = pd.read_csv(validation_filename, delimiter='*')
# validation_data.drop(columns=['tweets'],inplace=True)
validation_data = validation_data.values.tolist()
print(len(validation_data))
queue = Queue()
for img_file,hashtags in validation_data:
    queue.put(img_file)
ts = time()
procs = [Process(target=run_main, args=[i, queue]) for i in range(num_processes)]
for p in procs:
    p.start()
for p in procs:
    p.join()

print('Took {}s'.format(time() - ts))