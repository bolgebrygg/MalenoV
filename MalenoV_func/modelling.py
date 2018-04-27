# Make initial package imports
import keras
from keras.models import Sequential, Dense, Activation, Flatten, Dropout, Conv3D
from keras.layers.normalization import BatchNormalization

### ---- Make the model for the neural network ----
def make_model(cube_size = 65, num_channels = 1, num_classes = 2, opt = keras.optimizers.adam(lr=0.001)):
    #  This model is loosely built after that of Anders Waldeland (5 Convolutional layers
    #  and 2 fully connected layers with rectified linear and softmax activations)
    #  We have added drop out and batch normalization our selves, and experimented with multi-prediction
    #
    #  We also use the Adam optimizer with a given learning rate (Note that this is adapted later)
    model = Sequential()
    model.add(Conv3D(50, (5, 5, 5), padding='same', input_shape=(cube_size,cube_size,cube_size,num_channels), strides=(4, 4, 4), \
                     data_format="channels_last",name = 'conv_layer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding = 'same',name = 'conv_layer2'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer3'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer4'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer5'))
    model.add(Flatten())
    model.add(Dense(50,name = 'dense_layer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10,name = 'attribute_layer'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, name = 'pre-softmax_layer'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))


    # Compile the model with the desired loss, optimizer, and metric
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model
