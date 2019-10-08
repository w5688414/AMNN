import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import time
from tqdm import tqdm
from config import configs
class Evaluator(object):

    def __init__(self, model,
            data_path='preprocessed_data/',
            images_path='iaprtc12/',
            log_filename='data_parameters.log',
            test_data_filename='validation_data.txt',
            word_to_id_filename='word_to_id.p',
            id_to_word_filename='id_to_word.p',
            image_name_to_features_filename=None):
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
                                        image_name_to_features_filename[0])
        self.image_names_to_features_scene = h5py.File(data_path +
                                        image_name_to_features_filename[1])

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
        features_scene = self.image_names_to_features_scene[image_name]['image_features'][:]
        
        # print(features)
        image_input= np.concatenate((features,features_scene))
        # print(image_input)
        images_batch = np.zeros((1, 8192))
        images_batch[0,:]=image_input
        # print(images_batch)
        num=0
        list_word_id=[]
        predictions = self.model.predict(images_batch)
        # print(predictions)
        matrix=np.argsort(predictions[0])
        for id in reversed(matrix):
            num+=1
            word=self.id_to_word[id]
            print(word,end=" ")
            if(num==configs['num_hashtags']):
                break
            list_word_id.append(id)
            #images_path = '../dataset/images/'
        
        plt.imshow(plt.imread(self.images_path + image_name))
        plt.show()

    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_hashtags.txt'

        predicted_captions = open(dump_filename, 'w')

        image_names = self.test_data['image_names'].tolist()
        count=0
        for image_name in tqdm(image_names):
            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            features_scene = self.image_names_to_features_scene[image_name]\
                                            ['image_features'][:]
            image_input= np.concatenate((features,features_scene))
            images_batch = np.zeros((1, 8192))
            images_batch[0,:]=image_input

            num=0
            list_word_id=[]
            predictions = self.model.predict(images_batch)
            matrix=np.argsort(predictions[0])
            neural_caption=[]
            for id in reversed(matrix):
                num+=1
                word=self.id_to_word[id]
                # print(word,end=" ")
                if(num==configs['num_hashtags']+1):
                    break
                else:
                    neural_caption.append(word)
                list_word_id.append(id)
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['caption']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)

if __name__ == '__main__':
    from keras.models import load_model
    object_image_features_filename='vgg16_image_name_to_features.h5'
    scene_image_features_filename='vgg16_places_image_name_to_features.h5'
    list_image_feature=[]
    list_image_feature.append(object_image_features_filename)
    list_image_feature.append(scene_image_features_filename)
    # root_path = 'data/'
    # root_path='data/custom/'
    # image_path='/home/eric/data/social_images/'
    # root_path = 'data/NUS-WIDE/'
    # image_path='/home/eric/data/NUS-WIDE/image/'
    image_path='/home/eric/data/HARRISON/'
    root_path='data/HARRISON/'
    data_path = root_path + 'preprocessed_data/'
    # image_path='/home/eric/data/HARRISON/'
    
    
    model_filename = 'trained_model/vggbaseline/weights_final.hdf5'
    model = load_model(model_filename)
    # print(model.summary())
    evaluator = Evaluator(model, data_path, image_path,image_name_to_features_filename=list_image_feature)
    evaluator.write_captions()
    evaluator.display_caption()
