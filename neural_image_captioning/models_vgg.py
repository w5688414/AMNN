from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense
from keras.layers import BatchNormalization, Embedding, Activation, Reshape,Permute,Merge
from keras.layers.merge import Add,Dot
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers import merge
from keras.models import Sequential

def NIH(max_token_length, vocabulary_size, rnn='gru' ,num_image_features=2048,
        hidden_size=512, embedding_size=512, regularizer=1e-8):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    #  object image embedding
    object_image_input = Input(shape=(max_token_length, num_image_features),
                                                        name='image_object')
    object_image_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='object_image_embedding'))(object_image_input)
    object_image_dropout = Dropout(.5,name='object_image_dropout')(object_image_embedding)

     # scene image embedding
    scene_image_input = Input(shape=(max_token_length, num_image_features),
                                                        name='image_scene')
    scene_image_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='scene_image_embedding'))(scene_image_input)
    scene_image_dropout = Dropout(.5,name='scene_image_dropout')(scene_image_embedding)

    #object attention
    object_attention_network = LSTM(units=hidden_size,
                                return_sequences=True,
                                name='object_attention_recurrent_network')(object_image_input)
    object_attention_network = Activation('tanh')(object_attention_network)
    object_attention_network = TimeDistributed(Dense(1))(object_attention_network)
    object_attention_probs = Activation('softmax')(object_attention_network)
    object_attention_mul =merge([object_image_embedding, object_attention_probs],  mode='dot', dot_axes=(1, 1))
    object_attention_permute=Permute((2, 1))(object_attention_mul)

    #scene attention
    scene_attention_network = LSTM(units=hidden_size,
                                return_sequences=True,
                                name='scene_attention_recurrent_network')(scene_image_input)
    scene_attention_network = Activation('tanh')(scene_attention_network)
    scene_attention_network = TimeDistributed(Dense(1))(scene_attention_network)
    scene_attention_probs = Activation('softmax')(scene_attention_network)
    scene_attention_mul =merge([object_image_embedding, scene_attention_probs],  mode='dot', dot_axes=(1, 1))
    scene_attention_permute=Permute((2, 1))(scene_attention_mul)


    # language model
   # recurrent_inputs = [text_dropout, image_dropout]
    left_recurrent_inputs = [text_dropout, object_attention_permute]
    left_merged_input = Add()(left_recurrent_inputs)
    right_recurrent_inputs = [text_dropout, scene_attention_permute]
    right_merged_input = Add()(right_recurrent_inputs)
    if rnn == 'gru':
        left_recurrent_network = GRU(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                name='forward_recurrent_network')(left_merged_input)
        right_recurrent_network = GRU(units=hidden_size,
                                recurrent_regularizer=l2(regularizer),
                                kernel_regularizer=l2(regularizer),
                                bias_regularizer=l2(regularizer),
                                return_sequences=True,
                                go_backwards=True,
                                name='backword_recurrent_network')(right_merged_input)       
    # if rnn == 'lstm':
    #     recurrent_network = LSTM(units=hidden_size,
    #                             recurrent_regularizer=l2(regularizer),
    #                             kernel_regularizer=l2(regularizer),
    #                             bias_regularizer=l2(regularizer),
    #                             return_sequences=True,
    #                             name='recurrent_network')(merged_input)

    # elif rnn == 'gru':
    #     recurrent_network = GRU(units=hidden_size,
    #                             recurrent_regularizer=l2(regularizer),
    #                             kernel_regularizer=l2(regularizer),
    #                             bias_regularizer=l2(regularizer),
    #                             return_sequences=True,
    #                             name='recurrent_network')(merged_input)
    else:
        raise Exception('Invalid rnn name')
    merged_recur=merge([left_recurrent_network, right_recurrent_network], mode='sum')
    output = TimeDistributed(Dense(units=vocabulary_size,
                                    kernel_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(merged_recur)

    inputs = [text_input, object_image_input,scene_image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = NIH(16, 1024)
    plot_model(model, '../images/NIH.png')
