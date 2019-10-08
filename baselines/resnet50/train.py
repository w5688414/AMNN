from generator import Generator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from config import configs
from keras.utils import plot_model
from VGG import VGGNet
from measure import precision,recall, fmeasure
from resnet50 import ResNet50

batch_size=configs['batch_size']
num_epochs=configs['num_epochs']
data_path='data/HARRISON/preprocessed_data/'
# image_path='/home/eric/data/social_images/'
image_path='/home/eric/data/HARRISON/'
model_name='resnet50'
object_image_features_filename='resnet50_image_name_to_features.h5'
generator=Generator(data_path=data_path,batch_size=batch_size,image_features_filename=object_image_features_filename)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# model=VGGNet.vgg(input_shape=(224,224,3),classes=generator.VOCABULARY_SIZE)
model=ResNet50.resnet(input_shape=(configs['img_width'],configs['img_height'],configs['channel']), classes=generator.VOCABULARY_SIZE)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy",fmeasure,precision,recall])

print(model.summary())
plot_model(model,show_shapes=True,to_file=model_name+'.png')

save_path=os.path.join('trained_model',model_name)
if(not os.path.exists(save_path)):
    os.makedirs(save_path)
model_names = (os.path.join(save_path,'hashtag_weights.{epoch:02d}-{val_loss:.4f}.hdf5'))
model_checkpoint = ModelCheckpoint(filepath=model_names, 
                            monitor='val_loss',verbose=1, save_best_only=True, mode='max')
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
model.save(os.path.join(save_path,"weights_final.hdf5"))

from evaluate import Evaluator
import matplotlib.pyplot as plt


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


