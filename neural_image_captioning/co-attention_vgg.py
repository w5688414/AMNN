from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense,Flatten
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

    # object -> scene
    attention_network = LSTM(units=hidden_size,
                                return_sequences=True,
                                name='os_attention_recurrent_network')(scene_image_input)
    attention_network = TimeDistributed(Dense(1))(attention_network)
    attention_probs = Activation('softmax')(attention_network)
    attention_mul =merge([object_image_embedding, attention_probs],  mode='dot', dot_axes=(1, 1))
    attention_permute_object=Permute((2, 1))(attention_mul)

    # scene -> object
    attention_network = LSTM(units=hidden_size,
                                return_sequences=True,
                                name='so_attention_recurrent_network')(object_image_input)
    attention_network = TimeDistributed(Dense(1))(attention_network)
    attention_probs = Activation('softmax')(attention_network)
    attention_mul =merge([scene_image_embedding, attention_probs],  mode='dot', dot_axes=(1, 1))
    attention_permute_scene=Permute((2, 1))(attention_mul)

    # language model
   # recurrent_inputs = [text_dropout, image_dropout]
    merge_feature = merge([attention_permute_object, attention_permute_scene], mode='concat')
    merged_image_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='object_image_embedding'))(merge_feature)
    recurrent_inputs = [text_dropout, merged_image_embedding]
    merged_input = Add()(recurrent_inputs)
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

    inputs = [text_input, object_image_input,scene_image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = NIH(16, 1024)
    plot_model(model, '../images/NIH_coattention.png')
