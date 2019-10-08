from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense,concatenate,RepeatVector,Flatten
from keras.layers import BatchNormalization, Embedding, Activation, Reshape,Permute,Bidirectional
from keras.layers.merge import Add,Dot
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers import merge
from keras.layers import dot
from Attention import Attention

def NIC(max_token_length, vocabulary_size, rnn='lstm' ,num_image_features=2048,
        hidden_size=512, embedding_size=512, regularizer=1e-8):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    # image embedding  
    # encoder
    image_input = Input(shape=(max_token_length, num_image_features),
                                                        name='image')
    image_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='image_embedding'))(image_input)
    image_dropout = Dropout(.5,name='image_dropout')(image_embedding)

    #attention
    # attention_network = LSTM(units=hidden_size,
    #                             return_sequences=True,
    #                          name='attention_recurrent_network')(image_embedding)
    # attention_network = Activation('tanh')(attention_network)
    # attention_network = TimeDistributed(Dense(1))(attention_network)
    # attention_probs = Activation('softmax')(attention_network)
    # attention_mul=dot([image_embedding, attention_probs],axes=(1,1))
    # attention_permute=Permute((2, 1))(attention_mul)
    # # language model
    # recurrent_inputs = [text_dropout, attention_permute]

    recurrent_inputs = [text_dropout, image_dropout]
    merged_input = Add()(recurrent_inputs)
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

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

def bid_NIH(max_token_length, vocabulary_size, rnn='lstm' ,num_image_features=2048, 
            hidden_size=512, embedding_size=512, regularizer=1e-8):
    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    # image embedding
    image_input = Input(shape=(max_token_length, num_image_features),
                                                        name='image')
    image_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='image_embedding'))(image_input)
    image_dropout = Dropout(.5,name='image_dropout')(image_embedding)

    #attention
    attention_network = LSTM(units=hidden_size,
                                return_sequences=True,
                                name='attention_recurrent_network')(image_dropout)
    attention_network = Activation('tanh')(attention_network)
    attention_network = TimeDistributed(Dense(1))(attention_network)
    attention_probs = Activation('softmax')(attention_network)
    attention_mul =merge([image_embedding, attention_probs],  mode='dot', dot_axes=(1, 1))
    attention_permute=Permute((2, 1))(attention_mul)
    # language model
    # recurrent_inputs = [text_dropout, image_dropout]
    recurrent_inputs = [text_dropout, attention_permute]
    merged_input = Add()(recurrent_inputs)
    if rnn == 'lstm':
        recurrent_network = Bidirectional(LSTM(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='recurrent_network'))(merged_input)

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

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model



if __name__ == "__main__":
    from keras.utils import plot_model
    model_to_file='../images/NIC.png'
    model = NIC(22, 1000)
    # model_to_file='../images/bid_NIH.png'
    # model=bid_NIH(10,1024)
    plot_model(model,show_shapes=True,to_file=model_to_file)
