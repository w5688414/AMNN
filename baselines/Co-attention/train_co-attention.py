from generator import Generator
from co_attention import co_attention
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from config import configs
from keras.utils import plot_model

batch_size=configs['batch_size']
num_epochs=configs['num_epochs']
data_path='/home/eric/Documents/Hashtag-recommendation-for-social-images/baselines/data/preprocessed_data/'
object_image_features_filename='vgg16_image_name_to_features.h5'
generator=Generator(data_path=data_path,batch_size=batch_size,image_features_filename=object_image_features_filename)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model=co_attention(maxlen=configs['tweet_max_len'],vocab_size=configs['tweet_max_words'],embedding_dim=100,hashtag_size=generator.VOCABULARY_SIZE)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

print(model.summary())
plot_model(model,show_shapes=True,to_file='co_attention.png')
model_name='co_attention'
save_path=os.path.join('trained_model',model_name)
if(not os.path.exists(save_path)):
    os.makedirs(save_path)
model_names = (os.path.join(save_path,'hashtag_weights.{epoch:02d}-{val_loss:.4f}.hdf5'))
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint]
num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
print('Number of training samples:', num_training_samples)
print('Number of validation samples:', num_validation_samples)

history=model.fit_generator(generator=generator.flow('train'),
                    steps_per_epoch=num_training_samples // batch_size,
                    epochs=num_epochs,
                    callbacks=callbacks,
                    verbose=1,
                    validation_data=generator.flow('validation'),
                    validation_steps=num_validation_samples // batch_size)

from evaluate import Evaluator
import matplotlib.pyplot as plt

image_path='/home/eric/data/social_images/'
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

evaluator = Evaluator(model, data_path=data_path,
                      images_path=image_path,image_name_to_features_filename=object_image_features_filename)
evaluator.write_captions()
evaluator.display_caption()


