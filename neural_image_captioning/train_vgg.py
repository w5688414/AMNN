from __future__ import print_function
from evaluator_modified import Evaluator
from generator_vgg import Generator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from models_vgg import NIH
from data_manager_modified import DataManager
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
# config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5
# set_session(tf.Session(config=config))

num_epochs = 350
batch_size = 128
cnn_extractor='vgg16_places'
object_image_features_filename='vgg16_image_name_to_features.h5'
scene_image_features_filename='vgg16_places_image_name_to_features.h5'
root_path = '../datasets/HARRISON/'
captions_filename = root_path + 'image_tags.txt'
data_manager = DataManager(data_filename=captions_filename,
                            max_caption_length=50,
                            word_frequency_threshold=0,
                            extract_image_features=False,
                            cnn_extractor=cnn_extractor,
                            image_directory=root_path,
                            split_data=False,
                            dump_path=root_path + 'preprocessed_data/')

data_manager.preprocess()
print(data_manager.captions[0])
print(data_manager.word_frequencies[0:20])

list_image_feature=[]
list_image_feature.append(object_image_features_filename)
list_image_feature.append(scene_image_features_filename)
preprocessed_data_path = root_path + 'preprocessed_data/'
generator = Generator(data_path=preprocessed_data_path,
                      batch_size=batch_size,image_features_filename=list_image_feature)

num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
print('Number of training samples:', num_training_samples)
print('Number of validation samples:', num_validation_samples)

print(generator.VOCABULARY_SIZE)
print(generator.IMG_FEATS)
#generator.flow(mode='train')
#generator.format_to_one_hot("beach sea trip island japan")

model = NIH(max_token_length=generator.MAX_TOKEN_LENGTH,
            vocabulary_size=generator.VOCABULARY_SIZE,
            rnn='gru',
            num_image_features=generator.IMG_FEATS,
            hidden_size=128,
            embedding_size=128)
# print("test")
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

print(model.summary())
print('Number of parameters:', model.count_params())

training_history_filename = preprocessed_data_path + 'training_hashtag_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('../trained_models/hashtag/' +
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
                    steps_per_epoch=int(num_training_samples / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=int(num_validation_samples / batch_size))

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
plt.savefig('../results/acc_val_acc.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('../results/loss_val_loss.png')
# plt.show()
evaluator = Evaluator(model, data_path=preprocessed_data_path,
                      images_path=root_path,image_name_to_features_filename="vgg16_places_image_name_to_features.h5" )
evaluator.write_captions()
evaluator.display_caption()
