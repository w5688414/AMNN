from __future__ import print_function

from generator import Generator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from models import NIC
from data_manager import DataManager
from keras.utils import plot_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from evaluator import Evaluator
import matplotlib.pyplot as plt
from config import configs
from tensorflow import set_random_seed



set_random_seed(2019)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

num_epochs = 100
batch_size = 128
image_path='/home/eric/data/social_images/'
# cnn_extractor='resnet50'
# object_image_features_filename='resnet50_image_name_to_features.h5'

# cnn_extractor='vgg16'
# object_image_features_filename='vgg16_image_name_to_features.h5'
# cnn_extractor="resnet152"
# object_image_features_filename="resnet152_image_name_to_features.h5"
cnn_extractor="inception"
object_image_features_filename="inception_image_name_to_features.h5"
root_path = '../../datasets/image_text/'
captions_filename = root_path + 'image_text_data.txt'
data_manager = DataManager(data_filename=captions_filename,
                            max_caption_length=30,
                            word_frequency_threshold=2,
                            extract_image_features=False,
                            cnn_extractor=cnn_extractor,
                            image_directory=image_path,
                            split_data=True,
                            dump_path=root_path + 'preprocessed_data/')

data_manager.preprocess()
print(data_manager.captions[0])
print(data_manager.word_frequencies[0:20])

preprocessed_data_path = root_path + 'preprocessed_data/'
generator = Generator(data_path=preprocessed_data_path,
                      batch_size=batch_size,image_features_filename=object_image_features_filename)


num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
print('Number of training samples:', num_training_samples)
print('Number of validation samples:', num_validation_samples)

print(generator.VOCABULARY_SIZE)
print(generator.IMG_FEATS)
#generator.flow(mode='train')
#generator.format_to_one_hot("beach sea trip island japan")
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
# print("test")
# model.load_weights("../trained_models/hashtag/hashtag_weights.240-4.3745.hdf5")
# model.compile(loss='categorical_crossentropy',
#               optimizer = 'adam',
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

print(model.summary())
print('Number of parameters:', model.count_params())
plot_model(model,show_shapes=True,to_file='NIH.png')

training_history_filename = preprocessed_data_path + 'training_hashtag_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('trained_models/image_text/' +
               'hashtag_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [csv_logger, model_checkpoint, reduce_learning_rate]

history=model.fit_generator(generator=generator.flow(mode='train'),
                    steps_per_epoch=num_training_samples // batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=num_validation_samples // batch_size)
model.save("trained_models/image_text/weights_final.hdf5")
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
 # plt.show()
plt.savefig('./results/acc_val_acc.png')
plt.cla() # 清除axes
plt.clf() # 清除当前 figure 的所有axes
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./results/loss_val_loss.png')
# plt.show()

evaluator = Evaluator(model, data_path=preprocessed_data_path,
                      images_path=image_path,image_name_to_features_filename=object_image_features_filename)
evaluator.write_caption_after_train()
evaluator.display_caption_after_train()
