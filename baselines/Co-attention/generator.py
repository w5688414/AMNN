import pickle
import random

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from config import configs
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
        # self.MAX_TOKEN_LENGTH=10+2
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.VOCABULARY_SIZE = None
        self.word_to_id = None
        self.id_to_word = None
        self.BATCH_SIZE = batch_size

        self.load_dataset()
        self.load_vocabulary()
        self.load_image_features()
        self.tokenize_tweets()
        # self.get_embedding_matrix()

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

        print('Loading all tweet dataset...')
        complete_dataset = pd.read_table(
                                self.complete_filename,delimiter='*')
        complete_dataset = np.asarray(complete_dataset, dtype=str)

        self.complete_dataset=complete_dataset
    
    def tokenize_tweets(self,max_words=5000):
        """ 
            tokenize the whole datasets
        """
        tweets=self.complete_dataset[:,1]
        self.tokenizer = Tokenizer(num_words=configs['tweet_max_words'], lower=True)
        self.tokenizer.fit_on_texts(tweets)


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
        image_names = data[:,0].tolist()
        tweets=data[:,1]
        tweets_vec=self.get_tweet_features(tweets) #get tweet feature
        empty_batch = self.make_empty_batch()
        images_batch = empty_batch[0]
        tweets_batch = empty_batch[1]
        targets_batch=empty_batch[2]
        # print(len(image_names))
        # print(self.tweets_vec.shape)
        batch_counter = 0
        while True:
            for data_arg, image_name in enumerate(image_names):

                caption = data[data_arg,2]
                #print(caption)
                one_hot_caption = self.format_to_one_hot(caption)
                # print(images_batch.shape)
                image_feature=self.get_image_features(image_name)
                
                images_batch[batch_counter, :, :, :]   = self.get_image_features(image_name)
                tweets_batch[batch_counter,:]=tweets_vec[data_arg]
                targets_batch[batch_counter, :]  =one_hot_caption
                # print(data_arg)
                # print(data_arg)
                if batch_counter == self.BATCH_SIZE - 1:
                    # print(images_batch.shape)
                    yield_dictionary = self.wrap_in_dictionary( images_batch,
                                                                tweets_batch,
                                                                targets_batch)
                    yield yield_dictionary
                    empty_batch = self.make_empty_batch()
                    images_batch = empty_batch[0]
                    tweets_batch = empty_batch[1]
                    targets_batch=empty_batch[2]
                    batch_counter = 0

                batch_counter = batch_counter + 1

    def make_empty_batch(self):
        images_batch = np.zeros((self.BATCH_SIZE,7,7,512))
        tweets_batch=np.zeros((self.BATCH_SIZE,configs['tweet_max_len']))
        targets_batch = np.zeros((self.BATCH_SIZE,self.VOCABULARY_SIZE))
        return images_batch, tweets_batch,targets_batch

    def format_to_one_hot(self,caption):
        tokenized_caption = caption.split()
     #   tokenized_caption = [self.BOS] + tokenized_caption + [self.EOS]
        tokenized_caption = tokenized_caption
        #print(tokenized_caption)
        one_hot_caption = np.zeros((self.VOCABULARY_SIZE))
        word_ids = [self.word_to_id[word] for word in tokenized_caption
                        if word in self.word_to_id]
        # for sequence_arg, word_id in enumerate(word_ids):
        #     one_hot_caption[sequence_arg,word_id] = 1
        # print(word_ids)
        if(word_ids):
            word_id= random.choice(word_ids)
            one_hot_caption[word_id]=1
        return one_hot_caption

    def get_image_features(self, image_name):
        image_features = self.image_names_to_features[image_name]['image_features']
        image_input = np.zeros(( 1,7,7,512))
        # print(self.IMG_FEATS)
        # print(image_features.shape)
        # for i in range(self.MAX_TOKEN_LENGTH):
    
        image_input =  image_features
        return image_input

    def wrap_in_dictionary(self,image_features,tweet_feature,one_hot_target):

        return [{'image': image_features,
                'tweet':tweet_feature
                },
                {'output': one_hot_target}]


if __name__ == "__main__":
    batch_size = 1
    cnn_extractor='vgg16'
    object_image_features_filename='vgg16_image_name_to_features.h5'
    image_path='/home/eric/data/social_images/'

    root_path = 'data/'
    captions_filename = root_path + 'image_text_data.txt'

    preprocessed_data_path = root_path + 'preprocessed_data/'
    generator = Generator(data_path=preprocessed_data_path,
                        batch_size=batch_size,image_features_filename=object_image_features_filename)
    # print(next(generator.flow('train')))
    count=0
    while(True):
        batch_data=next(generator.flow('train'))
        print(batch_data[0]['tweet'].shape)
        print(batch_data[0]['image'].shape)
        print(batch_data[1]['output'].shape)
        batch_data=next(generator.flow('validation'))
        print(batch_data[0]['tweet'].shape)
        # print(count)
        # print(batch_data[0]['tweet'])
        # print(batch_data[0]['image'])
        # print(batch_data[1]['output'])
        # count+=1
        # if(count==1):
        #     break
        # print()



'''
class Generator():
    """ Data generator to the neural image captioning model (NIC).
    The flow method outputs a list of two dictionaries containing
    the inputs and outputs to the network.
    # Arguments:
        data_path = data_path to the preprocessed data computed by the
            Preprocessor class.
    """
    def __init__(self,data_path='/home/eric/Documents/Hashtag-recommendation-for-social-images/baselines/data/preprocessed_data',
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
                                            'image_feature.h5')
        else:
            self.image_features_filename = self.data_path + image_features_filename

        self.BATCH_SIZE = batch_size
        # self.dictionary = None
        # self.training_dataset = None
        # self.validation_dataset = None
        # self.image_names_to_features = None

        # data_logs = np.genfromtxt(self.data_path + 'data_parameters.log',
        #                           delimiter=' ', dtype='str')
        # data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))

        # self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        # self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        # self.BOS = str(data_logs['BOS:'])
        # self.EOS = str(data_logs['EOS:'])
        # self.PAD = str(data_logs['PAD:'])
        # self.VOCABULARY_SIZE = None
        # self.word_to_id = None
        # self.id_to_word = None
        # self.BATCH_SIZE = batch_size
        maxlen = 0
        x, img_x, y, maxlen=self.load_traindata(maxlen)
        valid_x, valid_img_x, valid_y, maxlen=self.load_testdata(maxlen)

        print("Train set size: ", len(x), len(img_x), len(y))
        print("Valid set size: ", len(valid_x), len(valid_img_x), len(valid_y))
        # print("Test set size: ", len(test_x), len(test_img_x), len(test_y))
        print("Max word len: ", maxlen)
        self.num_training_samples=len(x)
        self.num_validation_samples=len(valid_x)

        vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv =self.build_vocab(x, y, valid_x, valid_y)

        x, img_x, y = self.build_input_data(x, img_x, y, vocabulary, hashtagVoc, maxlen)
        valid_x, valid_img_x, valid_y = self.build_test_data(valid_x, valid_img_x, valid_y, vocabulary, hashtagVoc, maxlen)

        self.x=x
        self.img_x=img_x
        self.y=y

        self.valid_x=valid_x
        self.valid_img_x=valid_img_x
        self.valid_y=valid_y

        self.maxlen=maxlen

        self.vocabulary=vocabulary
        self.vocabulary_inv=vocabulary_inv
        self.hashtagVoc=hashtagVoc
        self.hashtagVoc_inv=hashtagVoc_inv

        self.load_image_features()

        # self.load_dataset()
        # self.load_vocabulary()
        # self.load_image_features()
        # print(len(hashtagVoc))
        # print(len(vocabulary_inv))
        self.VOCABULARY_SIZE = len(vocabulary)
        self.hashtag_size=len(hashtagVoc)
        # print(self.VOCABULARY_SIZE)
        # print(self.hashtag_size)

    def load_traindata(self,maxlen):
        print(self.training_filename)
        x = []
        img_x = []
        y = []
        index=0
        cur_hashtag_count = 0
        infile = open(self.training_filename)
        for line in tqdm(infile):
            index+=1
            if index%100000==0:
                print(index)
            line = line.strip().split('\t')
            hashtagList = line[2].split('||')
            cur_hashtag_count = len(hashtagList)
            tweet = line[1].split(' ')
            tweet = [w.rstrip() for w in tweet if w != '']
            maxlen = max(len(tweet), maxlen)
            for i in range(cur_hashtag_count):
                x.append(tweet)   #ｘ:一个tweet+一个image, y:一个hashtag
                img_x.append(line[0])
            for h in hashtagList:
                y.append(h)

        return x, img_x, y, maxlen
    
    def load_testdata(self,maxlen):
        print(self.validation_filename)
        x = []
        img_x = []
        y = []
        index = 0
        cur_hashtag_count = 0
        infile = open(self.validation_filename)
        for line in infile:
            line = line.strip().split('\t')
            hashtagList = line[2].split('||')
            tweet = line[1].split(' ')
            tweet = [w.rstrip() for w in tweet if w != '']
            maxlen = max(len(tweet), maxlen)
            # x.append(tweet)
            # y.append(hashtagList)
            cur_hashtag_count = len(hashtagList)

            for h in hashtagList:
                y.append(h)
            for i in range(cur_hashtag_count):
                x.append(tweet)   #ｘ:一个tweet+一个image, y:一个hashtag
                img_x.append(line[0])
            # img_x.append(line[0])
        return x, img_x, y, maxlen

    def build_vocab(self,x, y, valid_x, valid_y):
        vocabFileName = self.data_path+'vocabulary_full.pkl'

        if os.path.isfile(vocabFileName):
            print("loading vocabulary...")
            vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv = pickle.load(open(vocabFileName,'rb'),encoding='bytes')
    #         vocabulary_inv = vocabulary_inv[:500000]
    #         vocabulary = {x: i + 1 for i, x in enumerate(vocabulary_inv)}
        else:
            print("calculating vocabulary...")
            text = []
            for x_t in [x, valid_x]:
                for t in x_t:
                    for w in t:
                        text.append(w)

            print(len(text))
            word_counts = Counter(text)

            vocabulary_inv = [x[0] for x in word_counts.most_common()]
            vocabulary = {x: i + 1 for i, x in enumerate(vocabulary_inv)}

            hashtaglist = []
            for h in y:
                hashtaglist.append(h)
            for hlist in valid_y:
                for h in hlist:
                    hashtaglist.append(h)

            hashtag_count = Counter(hashtaglist)
            hashtagVoc_inv = [x[0] for x in hashtag_count.most_common() if x[1]>1]  #取频率大于1的hastag
            hashtagVoc = {x: i for i, x in enumerate(hashtagVoc_inv)}

            pickle.dump([vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv], open(vocabFileName, "wb"))
        
        return [vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv]
    
    def build_input_data(self,x, img_x, y, vocabulary, hashtagVoc, maxlen):
        x = np.asarray([[vocabulary[w] for w in t if w in vocabulary] for t in x])
        x = pad_sequences(x, maxlen=maxlen).astype(np.int32)
        img_x = np.asarray(img_x)
        # print(img_x)
        # print(len(img_x))
        
        y = np.asarray([hashtagVoc[h] for h in y], dtype=np.int32)

        return [x, img_x, y]

    def build_test_data(self,x, img_x, y, vocabulary, hashtagVoc, maxlen):
        x = np.asarray([[vocabulary[w] for w in t if w in vocabulary] for t in x])
        x = pad_sequences(x, maxlen=maxlen).astype(np.int32)
        img_x = np.asarray(img_x)
        # y = np.asarray([[hashtagVoc[h] for h in hlist] for hlist in y])
        y = np.asarray([hashtagVoc[h] for h in y], dtype=np.int32)

        return [x, img_x, y]
    
    def load_image_features(self):
        self.image_names_to_features = h5py.File(
                                        self.image_features_filename, 'r')

    # def flow(self, mode):
    #     if mode == 'train':
    #         image_names=self.img_x
    #         self.get_train_batch(image_names)
    #         #random.shuffle(data) #this is probably correct but untested 
    #     if mode == 'validation':
    #         image_names=self.valid_img_x
    #         self.get_test_batch(image_names)

        
    def get_train_batch(self):
        image_names=self.img_x
        batch_counter = 0
        x_batch = []
        img_x_batch = []
        y_batch = []
        while True:
            for data_arg, image_name in enumerate(image_names):
                img_data = self.image_names_to_features.get(image_name)
                np_data = np.array(img_data)
                img_x_batch.append(np_data)
                x_batch.append(self.x[data_arg])
                y_batch.append(self.y[data_arg])
                if batch_counter == self.BATCH_SIZE - 1:
                    yield_dictionary=self.wrap_in_dictionary(x_batch,img_x_batch,y_batch)
                    
                    yield yield_dictionary
                    x_batch = []
                    img_x_batch = []
                    y_batch = []
                    batch_counter = 0
                batch_counter = batch_counter + 1
    
    def get_test_batch(self):
        image_names=self.valid_img_x
        batch_counter = 0
        x_batch = []
        img_x_batch = []
        y_batch = []
        while True:
            for data_arg, image_name in enumerate(image_names):
                img_data = self.image_names_to_features.get(image_name)
                np_data = np.array(img_data)
                img_x_batch.append(np_data)
                x_batch.append(self.valid_x[data_arg])
                y_batch.append(self.valid_y[data_arg])
                if batch_counter == self.BATCH_SIZE - 1:
                    yield_dictionary=self.wrap_in_dictionary(x_batch,img_x_batch,y_batch)
                    
                    yield yield_dictionary
                    x_batch = []
                    img_x_batch = []
                    y_batch = []
                    batch_counter = 0
                batch_counter = batch_counter + 1




    def wrap_in_dictionary(self,tweets,
                           image_features,
                           hashtags):
        tweets=np.array(tweets)
        image_features=np.array(image_features)  
        hashtags=np.array(hashtags) 
        hashtags = np_utils.to_categorical(hashtags, self.hashtag_size)

        return [{'tweets': tweets,
                'image': image_features},
                {'hashtags': hashtags}]


'''