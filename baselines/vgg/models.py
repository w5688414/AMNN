from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import BatchNormalization

def VGG_baseline(hashtag_size=10):
    model = Sequential()
    model.add(Dense(4096,input_shape=(8192,),name='image'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(hashtag_size))
    model.add(Activation('sigmoid',name='output'))

    return model