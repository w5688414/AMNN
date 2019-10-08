from keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dense, Flatten,Input
)
from keras.models import Model, load_model
import os

class VGGNet():

    @staticmethod
    def vgg(input_shape,classes=100,weights="trained_model/vggnet.hdf5"):
        """Inference function for VGGNet

        y = vgg(X)

        Parameters
        ----------
        input_tensor : keras.layers.Input

        Returns
        ----------
        y : softmax output tensor
        """
        def two_conv_pool(x, F1, F2, name):
            x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

            return x

        def three_conv_pool(x, F1, F2, F3, name):
            x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(F3, (3, 3), activation=None, padding='same', name='{}_conv3'.format(name))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

            return x

        input_tensor = Input(shape=input_shape,name='image')
        net = input_tensor
        net = two_conv_pool(net, 64, 64, "block1")
        net = two_conv_pool(net, 128, 128, "block2")
        net = two_conv_pool(net, 256, 256, "block3")
        net = three_conv_pool(net, 512, 512, 512, "block4")

        net = three_conv_pool(net, 512, 512, 512, "block5")

        net = Flatten()(net)
        net = Dense(4096, activation='relu', name='fc1')(net)
        net = Dense(4096, activation='relu', name='fc2')(net)
        net = Dense(classes, activation='sigmoid', name='output')(net)
        model = Model(input_tensor, net, name='model')

        if os.path.isfile(weights):
            model.load_weights(weights)
            print("Model loaded")
        else:
            print("No model is found")

        return model
    
# img_width=128
# img_height=128
# charset_size=6941
# model=VGGNet.vgg(input_shape=(img_width,img_height,3),classes=charset_size)
# model.summary()
# train(model,"vgg")
