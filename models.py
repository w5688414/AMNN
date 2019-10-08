from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense,concatenate,RepeatVector,Flatten
from keras.layers import CuDNNGRU,CuDNNLSTM
from keras.layers import BatchNormalization, Embedding, Activation, Reshape,Permute,Bidirectional,Conv1D,GlobalMaxPool1D,MaxPooling1D,Conv2D
from keras.layers.merge import Add,Dot
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers import merge
from keras.layers import dot,add,multiply
import numpy as np
from keras.backend import expand_dims
from config import configs
from Attention import Attention



def NIC(max_token_length, vocabulary_size,tweet_max_len=300,tweet_max_words=5000 ,rnn='lstm' ,num_image_features=2048,
        hidden_size=512, embedding_size=512, regularizer=1e-8,embedding_weights=None):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    tweet_input = Input(shape=(tweet_max_len,), dtype='int32',name='tweet')
    if configs['include_tweet']:
        # tweet embedding
        if(configs["w2v_weights"]):
            embedding = Embedding(input_dim=tweet_max_words, output_dim=tweet_max_len,weights=[embedding_weights],trainable=False, mask_zero=False)(tweet_input)
        else:
            embedding = Embedding(input_dim=tweet_max_words, output_dim=tweet_max_len, mask_zero=False)(tweet_input)
        embedding_1 = Bidirectional(CuDNNLSTM(units=embedding_size, return_sequences=True),merge_mode='sum')(embedding)
        # conv=Conv1D(embedding_size, 7, padding='same', activation='relu', strides=1)(embedding)
        # max_pool=MaxPooling1D(2)(conv)
        # conv=Conv1D(embedding_size, 7, activation='relu', padding='same')(conv)
        # tweet_out=GlobalMaxPool1D()(conv)
        embedding_2 = Attention(tweet_max_len)(embedding_1)
        tweet_out = Dense(embedding_size, activation="relu")(embedding_2)
        # tweet_out = Activation('softmax')(tweet_out)
    # tweet_out = Bidirectional(LSTM(units=embedding_size, return_sequences=True))(embedding)
    # tweet_embedding= TimeDistributed(Dense(units=embedding_size,
    #                                     kernel_regularizer=l2(regularizer),
    #                                     name='text_new_embedding'))(tweet_out)
    # tweet_embedding = Activation('softmax')(tweet_embedding)
    image_input = Input(shape=(max_token_length, num_image_features),
                                                            name='image')
    if configs['include_image']:
        # image embedding  
        # encoder
       
        image_embedding = TimeDistributed(Dense(units=embedding_size,
                                            kernel_regularizer=l2(regularizer),
                                            name='image_embedding'))(image_input)
        image_feature = Dropout(.5,name='image_dropout')(image_embedding)

        # attention_probs=Dense(embedding_size ,activation="softmax")(image_embedding)
        # image_feature=multiply([image_embedding,attention_probs])
        # image_feature=Permute((2, 1))(attention_mul)

        ## attention
        attention_network = CuDNNLSTM(units=hidden_size,
                                    return_sequences=True,
                                name='attention_recurrent_network')(image_embedding)
        attention_network = Activation('tanh')(attention_network)
        attention_network = TimeDistributed(Dense(1))(attention_network)
        attention_probs = Activation('softmax')(attention_network)
        attention_mul=dot([image_embedding, attention_probs],axes=(1,1))
        image_feature=Permute((2, 1))(attention_mul)
        
  
    if(configs['include_image'] and configs['include_tweet']):
        recurrent_inputs = [tweet_out, image_feature]
        merged_input = Add(name="image_text")(recurrent_inputs)
        merged_input = Add()([merged_input,text_dropout])

        # attention_probs=Dense(embedding_size ,activation="softmax")(merged_input)
        # merged_input=multiply([merged_input,attention_probs])
        # image_feature=Permute((2, 1))(attention_mul)
    elif(configs['include_image']):
        # merged_input=Add()([merged_input, text_dropout])
        recurrent_inputs = [text_dropout, image_feature]
        merged_input = Add()(recurrent_inputs)
    elif(configs['include_tweet']):
        recurrent_inputs = [text_dropout, tweet_out]
        merged_input = Add()(recurrent_inputs)


    # merged_input=BatchNormalization()(merged_input)
   
    # merged_input = concatenate(recurrent_inputs,axis=2)


    # decoder
    if rnn == 'lstm':
        recurrent_network = LSTM(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='recurrent_network')(merged_input)

    elif rnn == 'gru':
        recurrent_network = GRU(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='recurrent_network')(merged_input)
    else:
        raise Exception('Invalid rnn name')

    output = TimeDistributed(Dense(units=vocabulary_size,
                                    kernel_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(recurrent_network)

    inputs = [text_input, image_input,tweet_input]
    model = Model(inputs=inputs, outputs=output)
    return model




if __name__ == "__main__":
    from keras.utils import plot_model
    model_to_file='NIC.png'
    model = NIC(22, 1000, tweet_max_len=configs['tweet_max_len'],
                        tweet_max_words=configs['tweet_max_words'],
                        include_image=configs['include_image'], 
                        include_tweet=configs['include_tweet'])
    print(model.summary())
    # model_to_file='../images/bid_NIH.png'
    # model=bid_NIH(10,1024)
    plot_model(model,show_shapes=True,to_file=model_to_file)
