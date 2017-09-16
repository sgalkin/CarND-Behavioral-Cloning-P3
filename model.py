import keras.layers as layers
import keras.initializers as initializers
from keras.models import Sequential

def model(input_shape, initializer, activation, out_activation, dropout, spatial_dropout):
    m = Sequential()

    # idenetity lambda layer just to land input shape there and make changing of following layers easier
    m.add(layers.Lambda(input_shape=input_shape, lambda x: x))
    
    m.add(layers.Cropping2D(cropping=((55, 25), (0, 0))))

    m.add(layers.Lambda(lambda x: 2*x/255.0 - 1.))

    m.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation=activation, kernel_initializer=initializer))
    m.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation=activation, kernel_initializer=initializer))
    m.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation=activation, kernel_initializer=initializer))

    if spatial_dropout:
        m.add(layers.SpatialDropout2D(spatial_dropout))
    
    m.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation=activation, kernel_initializer=initializer))
    m.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation=activation, kernel_initializer=initializer))
    m.add(layers.MaxPooling2D((3, 3), strides=(1, 1)))

    if dropout:
        m.add(layers.Dropout(dropout))

    m.add(layers.Flatten())

    m.add(layers.Dense(100, activation=activation, kernel_initializer=initializer))
    m.add(layers.Dense(50, activation=activation, kernel_initializer=initializer))
    m.add(layers.Dense(10, activation=activation, kernel_initializer=initializer))
    m.add(layers.Dense(1, activation=out_activation, kernel_initializer=initializer))

    return m
