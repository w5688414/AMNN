import pickle
import random

import h5py
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
from tqdm import tqdm
from config import configs
from gensim.models import KeyedVectors
import os

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


      
        self.complete_filename = data_path + 'complete_data.txt'   # used for tweets
      
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

        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.BOS = str(data_logs['BOS:'])
        self.EOS = str(data_logs['EOS:'])
        self.PAD = str(data_logs['PAD:'])
        self.VOCABULARY_SIZE = None
        self.word_to_id = None
        self.id_to_word = None
        self.BATCH_SIZE = batch_size

        self.load_dataset()
        self.load_vocabulary()
        self.load_image_features()
        self.tokenize_tweets()
        if(configs['w2v_weights']):
            self.get_embedding_matrix()

    def load_vocabulary(self):
        print('Loading vocabulary...')
        word_to_id = pickle.load(open(self.data_path + 'word_to_id.p', 'rb'))
        id_to_word = pickle.load(open(self.data_path + 'id_to_word.p', 'rb'))
        self.VOCABULARY_SIZE = len(word_to_id)
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def load_image_features(self):
        if(self.is_already_opened_in_write_mode(self.image_features_filename)):
            print("file already open,")
        self.image_names_to_features = h5py.File(
                                        self.image_features_filename, 'r')
    
    def is_already_opened_in_write_mode(self,filename):
        if os.path.exists(filename):
            try:
                f = open(filename, 'r')
                f.close()
            except IOError:
                return True
        return False

    def load_dataset(self):

        print('Loading training dataset...')
        train_data = pd.read_csv(self.training_filename, delimiter='*')
        # train_data = np.asarray(train_data,dtype=str)
        self.training_dataset = train_data

        print('Loading validation dataset...')
        validation_dataset = pd.read_csv(
                                self.validation_filename,delimiter='*')
        # validation_dataset = np.asarray(validation_dataset, dtype=str)
        self.validation_dataset = validation_dataset

        print('Loading all tweet dataset...')
        complete_dataset = pd.read_csv(
                                self.complete_filename,delimiter='*')
        # complete_dataset = np.asarray(complete_dataset, dtype=str)

        self.complete_dataset=complete_dataset
    
    def tokenize_tweets(self):
        """ 
            tokenize the whole datasets
        """
        tweets=self.complete_dataset["tweets"]
        self.tokenizer = Tokenizer(num_words=configs['tweet_max_words'], lower=True)
        self.tokenizer.fit_on_texts(tweets)
        
    
    def get_embedding_matrix(self):
        #load embeddings
        print('loading word embeddings...')
        w2v='wiki'

        if(w2v=='wiki'):
            f = codecs.open('/home/eric/data/jigsaw-toxic-comment-classification-challenge/wiki.simple.vec', encoding='utf-8')
        elif(w2v=='glove'):
            f=open('/home/eric/data/NLP/glove.6B/glove.6B.300d.txt')
        elif(w2v=='google_news'):
            print('Indexing word vectors')
            EMBEDDING_FILE='/home/eric/data/NLP/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
            word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)
            print('Found %s word vectors of word2vec' % len(word2vec.vocab))

        words_not_found = []
        embedding_matrix = np.zeros((configs['tweet_max_words'], configs['tweet_max_len']))
        print('preparing embedding matrix...')
        word_index = self.tokenizer.word_index
        nb_words = min(configs['tweet_max_words'], len(word_index))
        #embedding matrix
        if(w2v=='google_news'):
            for word, i in word_index.items():
                if word in word2vec.vocab:
                    embedding_matrix[i] = word2vec.word_vec(word)
                else:
                    words_not_found.append(word)
        else:
            embeddings_index = {}
            for line in tqdm(f):
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
            print('found %s word vectors' % len(embeddings_index))
            for word, i in word_index.items():
                embedding_vector = embeddings_index.get(word)
                if (embedding_vector is not None) and len(embedding_vector) > 0:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    words_not_found.append(word)

        
        self.embedding_matrix=embedding_matrix
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("sample words not found: ", np.random.choice(words_not_found, 10))


    def get_tweet_features(self,text_series):
        """
        transforms text data to feature_vectors that can be used in the ml model.
        tokenizer must be available.
        """
        sequences = self.tokenizer.texts_to_sequences(text_series)
        return pad_sequences(sequences, maxlen=configs['tweet_max_len'])

    def flow(self, mode):
        if mode == 'train':
            data = self.training_dataset
            #random.shuffle(data) #this is probably correct but untested 
        if mode == 'validation':
            data = self.validation_dataset
        image_names = data["image_names"].tolist()
        tweets=data["tweets"].tolist()
        tweets_vec=self.get_tweet_features(tweets) #get tweet feature
        empty_batch = self.make_empty_batch()
        captions_batch = empty_batch[0]
        images_batch = empty_batch[1]
        targets_batch = empty_batch[2]
        tweets_batch=empty_batch[3]
        # print(len(image_names))
        # print(self.tweets_vec.shape)
        batch_counter = 0
        while True:
            for data_arg, image_name in enumerate(image_names):

                caption = data["hashtags"].iloc[data_arg]
                # print(caption)
                one_hot_caption = self.format_to_one_hot(caption)
                captions_batch[batch_counter, :, :] = one_hot_caption
                targets_batch[batch_counter, :, :]  = self.get_one_hot_target(one_hot_caption)
                images_batch[batch_counter, :, :]   = self.get_image_features(image_name)
                # IndexError: index 4555 is out of bounds for axis 0 with size 4555
                tweets_batch[batch_counter,:]=tweets_vec[data_arg]
                # print(data_arg)
                if batch_counter == self.BATCH_SIZE - 1:
                    yield_dictionary = self.wrap_in_dictionary(captions_batch,
                                                                tweets_batch,
                                                                images_batch,
                                                                targets_batch)
                    
                    yield yield_dictionary
                    empty_batch = self.make_empty_batch()
                    captions_batch = empty_batch[0]
                    images_batch = empty_batch[1]
                    targets_batch = empty_batch[2]
                    tweets_batch=empty_batch[3]
                    batch_counter = 0

                batch_counter = batch_counter + 1

  

    def make_test_input(self,image_name=None):

        if image_name == None:
            image_name = random.choice(self.training_dataset["image_names"].tolist())

        one_hot_caption = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                        self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        one_hot_caption[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, :, :] = self.get_image_features(image_name)
        return one_hot_caption, image_features, image_name

    def make_empty_batch(self):
        captions_batch = np.zeros((self.BATCH_SIZE,self.MAX_TOKEN_LENGTH,
                                    self.VOCABULARY_SIZE))
        images_batch = np.zeros((self.BATCH_SIZE, self.MAX_TOKEN_LENGTH,
                                    self.IMG_FEATS))
        tweets_batch=np.zeros((self.BATCH_SIZE,configs['tweet_max_len']))
        targets_batch = np.zeros((self.BATCH_SIZE,self.MAX_TOKEN_LENGTH,
                                    self.VOCABULARY_SIZE))
        return captions_batch, images_batch, targets_batch,tweets_batch

    def format_to_one_hot(self,caption):
        tokenized_caption = caption.split()
     #   tokenized_caption = [self.BOS] + tokenized_caption + [self.EOS]
        tokenized_caption = [self.BOS] + tokenized_caption
        #print(tokenized_caption)
        one_hot_caption = np.zeros((self.MAX_TOKEN_LENGTH,
                                    self.VOCABULARY_SIZE))
        word_ids = [self.word_to_id[word] for word in tokenized_caption
                        if word in self.word_to_id]
        for sequence_arg, word_id in enumerate(word_ids):
            one_hot_caption[sequence_arg,word_id] = 1
        return one_hot_caption

    def get_image_features(self, image_name):
        image_features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
        image_input = np.zeros((self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        # print(self.IMG_FEATS)
        # print(image_features.shape)
        for i in range(self.MAX_TOKEN_LENGTH):
            image_input[i,:] =  image_features
        return image_input

    def get_one_hot_target(self,one_hot_caption):
        one_hot_target = np.zeros_like(one_hot_caption)
        one_hot_target[:-1, :] = one_hot_caption[1:, :]
        return one_hot_target

    def wrap_in_dictionary(self,one_hot_caption,
                           tweet_feature,
                           image_features,
                           one_hot_target):

        return [{'text': one_hot_caption,
                'image': image_features,
                'tweet':tweet_feature
                },
                {'output': one_hot_target}]


if __name__ == "__main__":
    batch_size = 128
    cnn_extractor='inception'
    object_image_features_filename='inception_image_name_to_features.h5'
    image_path='/home/eric/data/social_images/'

    root_path = '../../datasets/image_text/'
    captions_filename = root_path + 'image_text_data.txt'

    preprocessed_data_path = root_path + 'preprocessed_data/'
    generator = Generator(data_path=preprocessed_data_path,
                        batch_size=batch_size,image_features_filename=object_image_features_filename)
    # print(next(generator.flow('train')))
    count=0
    while(True):
        batch_data=next(generator.flow('train'))
        print(batch_data[0]['tweet'].shape)
        batch_data=next(generator.flow('validation'))
        print(batch_data[0]['tweet'].shape)
        print(count)
        count+=1
        # print()