# import cv2
# import numpy as np
# # name=r"data/test/5153849.jpg"
# # im = cv2.resize(cv2.imread(name), (448,448))

# index=[[1,2,3,4,5],[2,1,3,4,5]]
# arr_index=np.array(index)
# print(arr_index.shape) 

# from generator import *

# gen=Generator()


from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Flatten, Dropout, Reshape, Layer, \
    ActivityRegularization, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.layers import Input, Dense, Embedding, merge, Dropout
from keras.layers import concatenate
from keras.layers import dot
import numpy as np
from config import configs

def co_attention(maxlen,vocab_size,embedding_dim,hashtag_size):

    # embedding_dim =100
    # nb_epoch = 20
    # batch_size = 2

    feat_dim = 512
    w = 7
    num_region = 49

    tweet = Input(shape=(maxlen,), dtype='int32',name='tweet')
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, mask_zero=False)(
        tweet)
    lstm = LSTM(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim))(embedding)
    dropout = Dropout(0.5)(lstm)


    img = Input(shape=( w, w,feat_dim),name='image')
    img_reshape = Reshape(( w * w,feat_dim))(img)
    # img_permute = Permute((2, 1))(img_reshape)

    # text->img
    img_dense = TimeDistributed(Dense(embedding_dim))(img_reshape)

    # tweet_avg = AveragePooling1D(pool_length=maxlen)(dropout)
    tweet_avg = AveragePooling1D(pool_size=maxlen)(dropout)
    tweet_avg = Flatten()(tweet_avg)

    tweet_avg_dense = Dense(embedding_dim)(tweet_avg)
    tweet_repeat = RepeatVector(w * w)(tweet_avg_dense)
    # att_1 = merge([tweet_repeat, img_dense], mode='concat')
    att_1=concatenate([tweet_repeat, img_dense])
    att_1 = Activation('tanh')(att_1)
    att_1 = TimeDistributed(Dense(1))(att_1)
    att_1 = Activation('softmax')(att_1)

    att_1_pro = Flatten()(att_1)

    
    # img_new = merge([att_1, img_dense], mode='dot', dot_axes=(1, 1))
    img_new=dot([att_1, img_dense],axes=(1,1))
    # img->text
    img_new_dense = Dense(embedding_dim)(img_new)
    img_new_dense_pro = Flatten()(img_new_dense)
    img_new_repeat = RepeatVector(maxlen)(img_new_dense_pro)

    tweet_dense = TimeDistributed((Dense(embedding_dim)))(dropout)
    # att_2 = merge([img_new_repeat, tweet_dense], mode='concat')
    att_2=concatenate([img_new_repeat, tweet_dense])
    att_2 = Activation('tanh')(att_2)
    att_2 = Activation('softmax')(att_2)

    # tweet_new = merge([att_2, dropout], mode='dot', dot_axes=(1, 1))
    tweet_new=dot([att_2, dropout],axes=(1,1))

    img_new_tanh = Dense(embedding_dim, activation='tanh')(img_new)
    tweet_new_tanh = Dense(embedding_dim, activation='tanh')(tweet_new)
    img_new_tanh_pro = Flatten()(img_new_tanh)
    img_new_tanh_repeat = RepeatVector(200)(img_new_tanh_pro)
    # merge = merge([img_new_tanh_repeat, tweet_new_tanh], mode='concat')
    merge=concatenate([img_new_tanh_repeat, tweet_new_tanh])
    z = Dense(1, activation='sigmoid')(merge)

    merge = Flatten()(merge)
    output = Dense(hashtag_size, activation='softmax', name='output')(merge)

    model = Model(inputs=[tweet, img], outputs=output)

    return model


if __name__ == "__main__":
    from keras.utils import plot_model
    model = co_attention(maxlen=configs['tweet_max_len'],vocab_size=configs['tweet_max_words'],embedding_dim=100,hashtag_size=1001)
    plot_model(model, './co-attention.png',show_shapes=True)


