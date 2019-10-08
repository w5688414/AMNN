from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D

def VGG16_places():
    """
    Build the architecture of VGG-16.
    """
    img_input = Input(shape=(224, 224, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Block 6
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(205, activation='softmax', name='fc8')(x)

    return Model(img_input, x)

if __name__ == '__main__':
    model = VGG16()
    for layer in model.layers:
        print('The output shape of layer {} is {}'.format(
            layer.name, layer.output_shape[1:]))