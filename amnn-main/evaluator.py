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
import os
# from Attention import Attention

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
        self.full_data=pd.read_table(data_path+"complete_data.txt", sep='*')
        self.word_to_id = pickle.load(open(data_path +
                                           word_to_id_filename, 'rb'))
        self.id_to_word = pickle.load(open(data_path +
                                           id_to_word_filename, 'rb'))
        self.VOCABULARY_SIZE = len(self.word_to_id)
        self.image_names_to_features = h5py.File(data_path +
                                        image_name_to_features_filename)
        self.initialize_token()
        
    
    def initialize_token(self):
        complete_filename = self.data_path + 'complete_data.txt'
        print('Loading all tweet dataset...')
        complete_dataset = pd.read_table(complete_filename,delimiter='*')
        complete_dataset = np.asarray(complete_dataset, dtype=str)
        tweets=complete_dataset[:,1]
        self.tokenizer = Tokenizer(num_words=configs['tweet_max_words'], lower=True)
        self.tokenizer.fit_on_texts(tweets)

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
            tweets=line[0][1]
        else:
            image_name = image_file
        tweets=[tweets]

        sequences = self.tokenizer.texts_to_sequences(tweets)
        tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])

        print(image_name)
        features = self.image_names_to_features[image_name]['image_features'][:]
        print(features.shape)
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        print(self.BOS)
        num=0
        list_word_id=[]
        for word_arg in range(self.MAX_TOKEN_LENGTH - 1):
            if(configs['include_tweet'] and configs['include_image']):
                predictions = self.model.predict([text, image_features,tweet_vec])
            elif(configs['include_tweet']):
                predictions = self.model.predict([text,tweet_vec])
            elif(configs['include_image']):
                predictions = self.model.predict([text,image_features])
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
            elif(num==(configs['num_hashtags'])):
                break
        print()
        print(list_word_id)
            #images_path = '../dataset/images/'
        plt.imshow(plt.imread(self.images_path + image_name))
        plt.show()

    def display_caption_after_train(self, image_file=None, data_name=None):

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
            tweets=line[0][1]
        else:
            image_name = image_file
        tweets=[tweets]

        sequences = self.tokenizer.texts_to_sequences(tweets)
        tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])

        print(image_name)
        features = self.image_names_to_features[image_name]['image_features'][:]
        print(features.shape)
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        print(self.BOS)
        num=0
        list_word_id=[]
        for word_arg in range(self.MAX_TOKEN_LENGTH - 1):
     
            predictions = self.model.predict([text, image_features,tweet_vec])
  
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
            elif(num==(configs['num_hashtags'])):
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
        features = self.image_names_to_features[image_name]['image_features'][:]
        print(features.shape)
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        print(self.BOS)
        model= self.model
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        predictions =intermediate_layer_model.predict([text, image_features,tweet_vec])

        return predictions
    
    def feature_extraction(self,layer_name,image_file=None, data_name=None):
        image_names = self.full_data['image_names'].tolist()
        tweet_list=self.full_data['tweets'].tolist()
        features_output=[]
        f_out = h5py.File(os.path.join(self.data_path,"features.h5"), "a")
        batch_tweets=[]
        batch_image=[]
        batch_text=[]
        batch_image_names=[]
        for image_arg,image_name in tqdm(enumerate(image_names)):
            tweet=str(tweet_list[image_arg])
            sequences = self.tokenizer.texts_to_sequences([tweet])
            tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])
            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            image_features[0, 0, :] = features

            batch_image_names.append(image_name)
            batch_tweets.append(tweet_vec[0,:])
            batch_text.append(text[0,:,:])
            batch_image.append(image_features[0,:,:])
            # print(len(batch_text))
            if(len(batch_text)>=configs['batch_size']):
                model= self.model
                intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
                predictions =intermediate_layer_model.predict([batch_text, batch_image,batch_tweets])
                # print(predictions.shape)
                for i in range(predictions.shape[0]):
                    f_out.create_dataset(name=batch_image_names[i], data=predictions[i].flatten())

                batch_tweets=[]
                batch_image=[]
                batch_text=[]
                batch_image_names=[]
            # print(predictions.flatten().shape)
            # features_output.append(predictions.flatten())
            # f_out.create_dataset(name=image_name, data=predictions.flatten())
            # break
        if(len(batch_text)>=0):
            model= self.model
            intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            predictions =intermediate_layer_model.predict([batch_text, batch_image,batch_tweets])
            for i in range(predictions.shape[0]):
                f_out.create_dataset(name=batch_image_names[i], data=predictions[i].flatten())
        f_out.close()
        
        # for i in tqdm(range(len(image_names))):
            
            # break




    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_hashtags.txt'
            

        predicted_captions = open(dump_filename, 'w')
        image_names = self.test_data['image_names'].tolist()
        tweet_list=self.test_data['tweets'].tolist()
        count=0
        
        for image_arg,image_name in tqdm(enumerate(image_names)):
            count+=1
            
            tweet=str(tweet_list[image_arg])  # nan bug
            # print(tweet)
            sequences = self.tokenizer.texts_to_sequences([tweet])
            tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])

            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            image_features[0, 0, :] = features
            neural_caption = []
            num=0
            list_word_id=[]
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                if(configs['include_tweet'] and configs['include_image']):
                    predictions = self.model.predict([text, image_features,tweet_vec])
                elif(configs['include_tweet']):
                    predictions = self.model.predict([text,tweet_vec])
                elif(configs['include_image']):
                    predictions = self.model.predict([text,image_features])

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
                elif(num==(configs['num_hashtags']+1)):
                    break
                else:
                    neural_caption.append(word)
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['hashtags']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)

    def write_caption_after_train(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_hashtags.txt'

        predicted_captions = open(dump_filename, 'w')
        image_names = self.test_data['image_names'].tolist()
        tweet_list=self.test_data['tweets'].tolist()
        count=0
        
        for image_arg,image_name in tqdm(enumerate(image_names)):
            count+=1
            
            tweet=str(tweet_list[image_arg])  # nan bug
            # print(tweet)
            sequences = self.tokenizer.texts_to_sequences([tweet])
            tweet_vec=pad_sequences(sequences, maxlen=configs['tweet_max_len'])

            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            image_features[0, 0, :] = features
            neural_caption = []
            num=0
            list_word_id=[]
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                predictions = self.model.predict([text, image_features,tweet_vec])
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
                elif(num==(configs['num_hashtags']+1)):
                    break
                else:
                    neural_caption.append(word)
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['hashtags']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)


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

if __name__ == '__main__':
    from keras.models import load_model
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    # 
    root_path = '../../datasets/image_text/'
    data_path = root_path + 'preprocessed_data/'
    image_path='/home/eric/data/social_images/'
    model_filename = './trained_models/image_text/hashtag_weights.55-5.3486.hdf5'
    # model_filename='../hashtag_weights.61-5.0035.hdf5'
    object_image_features_filename="inception_image_name_to_features.h5"
    model = load_custom_model(root_path,object_image_features_filename,model_filename)
    
    # model = load_model(model_filename,custom_objects={"Attention":Attention})
    # model = load_model(model_filename)
    print(model.summary())
    # vgg16_image_name_to_features
    evaluator = Evaluator(model, data_path, image_path,image_name_to_features_filename=object_image_features_filename)
    # evaluator.write_caption_after_train()
    # evaluator.display_caption_after_train()
    imddle_layer_output=evaluator.get_layer_output('image_text')
    print(imddle_layer_output.shape)
    evaluator.feature_extraction('image_text')
