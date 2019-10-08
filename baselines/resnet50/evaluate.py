import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import time
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config import configs
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from measure import *

class Evaluator(object):

    def __init__(self, model,
            data_path='preprocessed_data/',
            images_path='iaprtc12/',
            log_filename='data_parameters.log',
            test_data_filename='validation_data.txt',
            word_to_id_filename='word_to_id.p',
            id_to_word_filename='id_to_word.p',
            image_name_to_features_filename='inception_image_name_to_features.h5'):
        self.model = model
        self.data_path = data_path
        self.images_path = images_path
        self.log_filename = log_filename
        data_logs = self._load_log_file()
        self.test_data = pd.read_table(data_path +
                                       test_data_filename, sep='*')
        self.word_to_id = pickle.load(open(data_path +
                                           word_to_id_filename, 'rb'))
        self.id_to_word = pickle.load(open(data_path +
                                           id_to_word_filename, 'rb'))
        self.VOCABULARY_SIZE = len(self.word_to_id)
        self.image_names_to_features = h5py.File(data_path +
                                        image_name_to_features_filename)
        
    

    def _load_log_file(self):
        data_logs = np.genfromtxt(self.data_path + 'data_parameters.log',
                                  delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))
        return data_logs


    def display_caption(self, image_file=None, data_name=None):

        if data_name == 'ad_2016':
            test_data = self.test_data[self.test_data['image_names'].\
                                            str.contains('ad_2016')]
        elif data_name == 'iaprtc12':
            test_data = self.test_data[self.test_data['image_names'].\
                                            str.contains('iaprtc12')]
        else:
            test_data = self.test_data
        # print(test_data)

        if image_file == None:
            line=np.array(test_data.sample(1))
            # image_name = np.asarray(test_data.sample(1))[0][0]
            image_name = line[0][0]
        else:
            image_name = image_file

        print(image_name)
        features = self.image_names_to_features[image_name]['image_features']
        print(features.shape)


        image_features = np.zeros((1, 224,224,3))
        image_features = features
        # print(self.BOS)
        num=0
        list_word_id=[]
        predictions = self.model.predict([image_features])
        print(predictions)
        matrix=np.argsort(predictions[0])
        for id in reversed(matrix):
            word=self.id_to_word[id]
            print(word,end=" ")
            if(num==configs['num_hashtags']):
                break
            list_word_id.append(id)
            
            
            num+=1
        print()
        print(list_word_id)
            #images_path = '../dataset/images/'
        plt.imshow(plt.imread(self.images_path + image_name))
        plt.show()



    def get_layer_output(self,layer_name,image_file=None, data_name=None):
        if data_name == 'ad_2016':
            test_data = self.test_data[self.test_data['image_names'].\
                                            str.contains('ad_2016')]
        elif data_name == 'iaprtc12':
            test_data = self.test_data[self.test_data['image_names'].\
                                            str.contains('iaprtc12')]
        else:
            test_data = self.test_data
        
        if image_file == None:
            line=np.array(test_data.sample(1))
            # image_name = np.asarray(test_data.sample(1))[0][0]
            image_name = line[0][0]
            tweets=line[0][1]
        else:
            image_name = image_file

        tweets=[tweets]
        sequences = self.tokenizer.texts_to_sequences(tweets)
        tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])

        print(image_name)
        features = self.image_names_to_features[image_name]['image_features']
        print(features.shape)
 
        image_features = np.zeros((1, 224,224,3))
        image_features = features
        model= self.model
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        predictions =intermediate_layer_model.predict([text, image_features,tweet_vec])

        return predictions



    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_hashtags.txt'

        predicted_captions = open(dump_filename, 'w')
        image_names = self.test_data['image_names'].tolist()
        count=0
        
        for image_arg,image_name in tqdm(enumerate(image_names)):
            count+=1
            # print(image_name)
            features = self.image_names_to_features[image_name]['image_features']
            # print(features.shape)
            # image_features = np.zeros((1, 224,224,3 ))
            image_features = features
        # print(self.BOS)
            num=0
            list_word_id=[]
            neural_caption = []
            predictions = self.model.predict([image_features])
            # print(predictions)
            matrix=np.argsort(predictions[0])
            for id in reversed(matrix):
                word=self.id_to_word[id]
                # print(word,end=" ")

                if(num==configs['num_hashtags']):
                    break
                else:
                    neural_caption.append(word)

                list_word_id.append(id)
            
                num+=1
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['caption']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)



if __name__ == '__main__':
    from keras.models import load_model
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    # 
    # root_path = 'data/'
    root_path='data/HARRISON/'
    data_path = root_path + 'preprocessed_data/'
    # image_path='/home/eric/data/social_images/'
    image_path='/home/eric/data/HARRISON/'
    model_filename = './trained_model/resnet50/hashtag_weights.38-0.1451.hdf5'
    # model_filename='../hashtag_weights.61-5.0035.hdf5'
    object_image_features_filename="resnet50_image_name_to_features.h5"
    # model = load_custom_model(root_path,object_image_features_filename,model_filename)
    
    # model = load_model(model_filename,custom_objects={"Attention":Attention})
    model = load_model(model_filename,custom_objects={"fmeasure":fmeasure,"precision":precision,"recall":recall})
    print(model.summary())
    # vgg16_image_name_to_features
    evaluator = Evaluator(model, data_path, image_path,image_name_to_features_filename=object_image_features_filename)
    evaluator.write_captions()
    evaluator.display_caption()
    # imddle_layer_output=evaluator.get_layer_output('add_2')
    # print(imddle_layer_output)
