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
import multiprocessing
from multiprocessing import Process, Queue
from time import time, sleep
from keras.models import load_model
import tensorflow as tf
print("Number of cpu : ", multiprocessing.cpu_count())
num_processes= 1
from config import configs
from evaluator import Evaluator
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model




def write_to_file(filename,words, file='predicted_hashtags.txt'):
	f = open(file, 'a')
	f.write(filename+"*")
	list_hashtag=[]
	for item in words:
		list_hashtag.append(item[0])
		# f.write(str(item[0])+' ')
	f.write(' '.join(list_hashtag))
	f.write('\n')


def calculate(img_file,evaluator,layer_name):
    tweet=tweet_dict[img_file]
    sequences = evaluator.tokenizer.texts_to_sequences([tweet])
    tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])
    features = evaluator.image_names_to_features[img_file]\
                                            ['image_features'][:]
    text = np.zeros((1, evaluator.MAX_TOKEN_LENGTH, evaluator.VOCABULARY_SIZE))
    begin_token_id = evaluator.word_to_id[evaluator.BOS]
    text[0, 0, begin_token_id] = 1
    image_features = np.zeros((1, evaluator.MAX_TOKEN_LENGTH,
                                                evaluator.IMG_FEATS))
    image_features[0, 0, :] = features
    model= evaluator.model
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    predictions =intermediate_layer_model.predict([text, image_features,tweet_vec])
    print(predictions)
    print(predictions.shape)


def run_main(proc_num, queue):
    print(proc_num, "Start loop")
    while True:
        if queue.empty():
            break
        item = queue.get()
        calculate(item,evaluator,'image_text')


def load_custom_model(root_path,object_image_features_filename,model_filename):
    from Attention import Attention
    from models import NIC
    from generator import Generator

    preprocessed_data_path = root_path + 'preprocessed_data/'
    generator = Generator(data_path=preprocessed_data_path,
                      batch_size=configs["batch_size"] ,image_features_filename=object_image_features_filename)
    if(configs['w2v_weights']):
        embedding_weights=generator.embedding_matrix
    else:
        embedding_weights=None
    model =  NIC(max_token_length=generator.MAX_TOKEN_LENGTH,
            vocabulary_size=generator.VOCABULARY_SIZE,
            tweet_max_len=configs['tweet_max_len'],
            tweet_max_words=configs['tweet_max_words'],
            rnn='gru',
            num_image_features=generator.IMG_FEATS,
            hidden_size=256,
            embedding_size=128,
            embedding_weights=embedding_weights)
    model.load_weights(model_filename)
    # model.load_model(model_filename)
    return model


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

root_path = '../../datasets/image_text/'
data_path = root_path + 'preprocessed_data/'
image_path='/home/eric/data/social_images/'
model_filename = './trained_models/image_text/hashtag_weights.59-5.3446.hdf5'
# model_filename='../hashtag_weights.61-5.0035.hdf5'
object_image_features_filename="inception_image_name_to_features.h5"
model = load_custom_model(root_path,object_image_features_filename,model_filename)
    
# model = load_model(model_filename,custom_objects={"Attention":Attention})
# model = load_model(model_filename)
print(model.summary())
# vgg16_image_name_to_features
evaluator = Evaluator(model, data_path, image_path,image_name_to_features_filename=object_image_features_filename)	
image_names = evaluator.full_data['image_names'].tolist()
tweet_list=evaluator.full_data['tweets'].tolist()
tweet_dict={}
for image_arg,image_name in tqdm(enumerate(image_names)):
    tweet=str(tweet_list[image_arg])
    tweet_dict[image_name]=tweet


if __name__ == "__main__":
    ts = time()
    seconds=0
    queue = Queue()
    for img_file in image_names:
        queue.put(img_file)

    print(queue.qsize())
    procs = [Process(target=run_main, args=[i, queue]) for i in range(num_processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print('Took {}s'.format(time() - ts))
