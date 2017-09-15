import keras.layers as layers
import keras.initializers as initializers
from keras.models import Sequential

def model(input_shape, initializer='truncated_normal', activation='elu', out_activation='linear'):
    m = Sequential()
    
    m.add(layers.Cropping2D(input_shape=input_shape, cropping=((55, 25), (0, 0))))
    
    m.add(layers.Lambda(lambda x: 2*x/255.0 - 1.))
    
    m.add(layers.Conv2D(24, (7, 7), strides=(2, 2), activation=activation, kernel_initializer=initializer))
    m.add(layers.Conv2D(36, (7, 7), strides=(2, 2), activation=activation, kernel_initializer=initializer))
    m.add(layers.Conv2D(48, (7, 7), strides=(2, 2), activation=activation, kernel_initializer=initializer))

    m.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation=activation, kernel_initializer=initializer))
    m.add(layers.Conv2D(72, (3, 3), strides=(1, 1), activation=activation, kernel_initializer=initializer))

    m.add(layers.Dropout(0.5))

    m.add(layers.Flatten())

    m.add(layers.Dense(300, activation=activation, kernel_initializer=initializer))
    m.add(layers.Dense(50, activation=activation, kernel_initializer=initializer))
    m.add(layers.Dense(10, activation=activation, kernel_initializer=initializer))
    m.add(layers.Dense(1, activation=out_activation, kernel_initializer=initializer))

    return m
