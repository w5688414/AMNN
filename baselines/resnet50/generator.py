import pickle
import random

import h5py
import numpy as np
import pandas as pd

class Generator():
    """ Data generator to the neural image captioning model (NIC).
    The flow method outputs a list of two dictionaries containing
    the inputs and outputs to the network.
    # Arguments:
        data_path = data_path to the preprocessed data computed by the
            Preprocessor class.
    """
    def __init__(self,data_path='preprocessed_data/',
                 training_filename=None,
                 validation_filename=None,
                 image_features_filename=None,
                 batch_size=100):

        self.data_path = data_path
        if training_filename == None:
            self.training_filename = data_path + 'training_data.txt'
        else:
            self.training_filename = self.data_path + training_filename


        if validation_filename == None:
            self.validation_filename = data_path + 'validation_data.txt'
        else:
            self.validation_filename = self.data_path + validation_filename

        if image_features_filename == None:
            self.image_features_filename = (data_path +
                                            'inception_image_name_to_features.h5')
        else:
            self.image_features_filename = self.data_path + image_features_filename


        self.dictionary = None
        self.training_dataset = None
        self.validation_dataset = None
        self.image_names_to_features = None

        data_logs = np.genfromtxt(self.data_path + 'data_parameters.log',
                                  delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))

        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) 
        # self.MAX_TOKEN_LENGTH=10+2
        self.IMG_FEATS = (1,224,224,3)
        self.VOCABULARY_SIZE = None
        self.word_to_id = None
        self.id_to_word = None
        self.BATCH_SIZE = batch_size

        self.load_dataset()
        self.load_vocabulary()
        self.load_image_features()

    def load_vocabulary(self):
        print('Loading vocabulary...')
        word_to_id = pickle.load(open(self.data_path + 'word_to_id.p', 'rb'))
        id_to_word = pickle.load(open(self.data_path + 'id_to_word.p', 'rb'))
        self.VOCABULARY_SIZE = len(word_to_id)
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
    
    def load_image_features(self):
        self.image_names_to_features = h5py.File(
                                        self.image_features_filename, 'r')


    def load_dataset(self):

        print('Loading training dataset...')
        train_data = pd.read_table(self.training_filename, delimiter='*')
        train_data = np.asarray(train_data,dtype=str)
        self.training_dataset = train_data

        print('Loading validation dataset...')
        validation_dataset = pd.read_table(
                                self.validation_filename,delimiter='*')
        validation_dataset = np.asarray(validation_dataset, dtype=str)
        self.validation_dataset = validation_dataset


    def flow(self, mode):
        if mode == 'train':
            data = self.training_dataset
            #random.shuffle(data) #this is probably correct but untested 
        if mode == 'validation':
            data = self.validation_dataset
        image_names = data[:,0].tolist()
        empty_batch = self.make_empty_batch()
        images_batch = empty_batch[0]
        targets_batch = empty_batch[1]

        batch_counter = 0
        while True:
            for data_arg, image_name in enumerate(image_names):

                caption = data[data_arg,1]
                #print(caption)
                one_hot_caption = self.format_to_one_hot(caption)
            
                targets_batch[batch_counter, :]  = one_hot_caption
                images_batch[batch_counter, :, :]   = self.get_image_features(
                                                            image_name)

                if batch_counter == self.BATCH_SIZE - 1:
                    yield_dictionary = self.wrap_in_dictionary(images_batch,
                                                                targets_batch)
                    yield yield_dictionary

                    empty_batch = self.make_empty_batch()
                   
                    images_batch = empty_batch[0]
                    targets_batch = empty_batch[1]
                    batch_counter = 0

                batch_counter = batch_counter + 1

    def make_test_input(self,image_name=None):

        if image_name == None:
            image_name = random.choice(self.training_dataset[:, 0].tolist())

        one_hot_caption = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                        self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        one_hot_caption[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, :, :] = self.get_image_features(image_name)
        return one_hot_caption, image_features, image_name

    def make_empty_batch(self):
    
        images_batch = np.zeros((self.BATCH_SIZE, 224,224,3))
        targets_batch = np.zeros((self.BATCH_SIZE,self.VOCABULARY_SIZE))
        return  images_batch , targets_batch

    def format_to_one_hot(self,caption):
        tokenized_caption = caption.split()
     #   tokenized_caption = [self.BOS] + tokenized_caption + [self.EOS]
        tokenized_caption = tokenized_caption
        #print(tokenized_caption)
        one_hot_caption = np.zeros((self.VOCABULARY_SIZE))
        word_ids = [self.word_to_id[word] for word in tokenized_caption
                        if word in self.word_to_id]
        for sequence_arg, word_id in enumerate(word_ids):
            one_hot_caption[word_id] = 1
        return one_hot_caption

    def get_image_features(self, image_name):
        image_features = self.image_names_to_features[image_name]\
                                            ['image_features']
       
        return image_features

    def get_one_hot_target(self,one_hot_caption):
        one_hot_target = np.zeros_like(one_hot_caption)
        one_hot_target[:-1, :] = one_hot_caption[1:, :]
        return one_hot_target

    def wrap_in_dictionary(self,image_features,
                           one_hot_target):

        return [{'image': image_features},
                {'output': one_hot_target}]

if __name__ == "__main__":
    batch_size = 2
    cnn_extractor='vgg16'
    object_image_features_filename='vgg16_image_name_to_features.h5'
    image_path='/home/eric/data/social_images/'

    root_path = 'data/'
    captions_filename = root_path + 'image_data.txt'

    preprocessed_data_path = root_path + 'preprocessed_data/'
    generator = Generator(data_path=preprocessed_data_path,
                        batch_size=batch_size,image_features_filename=object_image_features_filename)
    # print(next(generator.flow('train')))
    count=0
    while(True):
        batch_data=next(generator.flow('train'))
        print(batch_data[0]['image'])
        print(batch_data[1]['output'])
        batch_data=next(generator.flow('validation'))


