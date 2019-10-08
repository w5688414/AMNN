import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import time
from losses import categorical_focal_loss
from tqdm import tqdm
from keras.models import Model

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
        self.BOS = str(data_logs['BOS:'])
        self.EOS = str(data_logs['EOS:'])
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
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
            image_name = np.asarray(test_data.sample(1))[0][0]
        else:
            image_name = image_file
        print(image_name)
        features = self.image_names_to_features[image_name]['image_features'][:]
        print(features.shape)
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        # image_features[0, 0, :] = features
        for i in range(self.MAX_TOKEN_LENGTH):
            image_features[0,i,:] =  features
        print(image_features)
        print(self.BOS)
        num=0
        list_word_id=[]
        for word_arg in range(self.MAX_TOKEN_LENGTH - 1):
            predictions = self.model.predict([text, image_features])
            matrix=np.argsort(predictions[0, word_arg, :])
            word_id=0
            # print(matrix)
            word_id=matrix[-1]
            for id in reversed(matrix):
                if id not in list_word_id:
                    # print(id)
                    target_word_id=id
                    list_word_id.append(id)
                    break
            # word_id = np.argmax(predictions[0, word_arg, :])
            # print(np.argmax(predictions[0, word_arg, :]))
            # list_wordid=list_wordid.append(word_id)
            next_word_arg = word_arg + 1
            text[0, next_word_arg, target_word_id] = 1
            # print(text)
            word = self.id_to_word[target_word_id]
            print(word,end=" ")
            num+=1
            if word == self.EOS:
                break
            elif(num==5):
                break
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
        # print(test_data)

        if image_file == None:
            image_name = np.asarray(test_data.sample(1))[0][0]
        else:
            image_name = image_file
        print(image_name)
        features = self.image_names_to_features[image_name]['image_features'][:]
        print(features.shape)
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        for i in range(self.MAX_TOKEN_LENGTH):
            image_features[0,i,:] =  features
        print(image_features)
        print(self.BOS)
        model= self.model
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        predictions =intermediate_layer_model.predict([text, image_features])

        return predictions

    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_hashtags.txt'

        predicted_captions = open(dump_filename, 'w')
        image_names = self.test_data['image_names'].tolist()
        count=0
        for image_name in tqdm(image_names):
            count+=1
            # if(count%1000==0):
            #     print(count)
            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            for i in range(self.MAX_TOKEN_LENGTH):
                image_features[0,i,:] =  features
            # image_features[0, 0, :] = features
            neural_caption = []
            num=0
            list_word_id=[]
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                predictions = self.model.predict([text, image_features])
                matrix=np.argsort(predictions[0, word_arg, :])
                word_id=0
                # print(matrix)
                word_id=matrix[-1]
                for id in reversed(matrix):
                    if id not in list_word_id:
                        # print(id)
                        target_word_id=id
                        list_word_id.append(id)
                        break                
                # word_id = np.argmax(predictions[0, word_arg, :])
                next_word_arg = word_arg + 1
                text[0, next_word_arg, target_word_id] = 1
                word = self.id_to_word[target_word_id]
                num+=1
                if word == '<E>':
                    break
                elif(num==6):
                    break
                else:
                    neural_caption.append(word)
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['caption']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)

if __name__ == '__main__':
    from keras.models import load_model
    # 
    # root_path = '../datasets/HARRISON/'
    root_path='../datasets/NUS-WIDE/'
    data_path = root_path + 'preprocessed_data/'
    # images_path = root_path
    images_path='/home/eric/data/NUS-WIDE/image/'
    # model_filename = '../trained_models/hashtag/xception.hdf5'
    model_filename='../trained_models/nuswide/hashtag_weights.51-4.5498.hdf5'
    # model_filename='../hashtag_weights.61-5.0035.hdf5'
    object_image_features_filename="inception_image_name_to_features.h5"
    model = load_model(model_filename)
    print(model.summary())
    # vgg16_image_name_to_features
    evaluator = Evaluator(model, data_path, images_path,image_name_to_features_filename=object_image_features_filename)
    evaluator.write_captions()
    evaluator.display_caption()


    # imddle_layer_output=evaluator.get_layer_output('attention_recurrent_network')
    # print(imddle_layer_output)
