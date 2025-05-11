from keras.layers import Dense, Dropout, Conv2D, Input, MaxPool2D, Flatten, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from tensorflow.keras import regularizers

import warnings
warnings.filterwarnings("ignore")

def proposed_model(nb_classes=5, input_h=128, input_w=128):
    input_shape = (input_h, input_w, 3)

    inputs = Input(input_shape)

    # layer 1
    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # layer3
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer4
    x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # layer5
    x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # layer6
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer7
    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # layer 9 - Global Average Pooling instead of MaxPool
    x = GlobalAveragePooling2D()(x)

    # layer 10 - Dense layers
    x = Dense(2048, kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # output layer
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs, x)
    return model

if __name__ == '__main__':
    cnn_model = proposed_model(nb_classes=5)
    print(cnn_model.summary())
