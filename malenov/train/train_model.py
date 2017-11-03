import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv3D
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization

# Make the network structure and outline, and train it
def train_model(segy_obj,class_array,num_classes,cube_incr,inp_res = np.float64,\
                num_bunch = 10,num_epochs = 100,num_examples = 10000,batch_size = 32,\
                opt_patience = 5, data_augmentation=False,num_channels = 1,\
                keras_model = None,write_out = False,write_location = 'default_write'):
    # segy_obj: Object returned from the segy_decomp function
    # class_array: numpy array of class adresses and type, returned from the convert function
    # num_classes: number of destinct classes we are training on
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # num_bunch: number of times we draw a new ensemble of training data and train on it
    # num_epochs: number of epochs we train on a given ensemble of training data
    # num_examples: number of examples we draw in an ensemble
    # batch_size: number of mini-batches we go through at a time from the number of examples
    # opt_patience: epochs that can pass without improvement in accuracy before the system breaks the loop
    # data_augmentation: boolean which determines whether or not to apply augmentation on the examples
    # num_channels: number of segy-cubes we have imported simultaneously
    # keras_model: existing keras model to be improved if the user wants to improve and not create a new model
    # write_out: boolean; save the trained model to disk or not,
    # write_location: desired location on the disk for the model to be saved

    # Check if the user wants to make a new model, or train an existing input model
    if keras_model == None:
        # Begin setting up model architecture and parameters
        cube_size = 2*cube_incr+1

        #  This model is loosely built after that of Anders Waldeland (5 Convolutional layers
        #  and 2 fully connected layers with rectified linear and softmax activations)
        # We have added drop out and batch normalization our selves, and experimented with multi-prediction
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

        # initiate the Adam optimizer with a given learning rate (Note that this is adapted later)
        opt = keras.optimizers.adam(lr=0.001)

        # Compile the model with the desired loss, optimizer, and metric
        model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    else:
        # Define the model we are performing training on as the input model
        model = keras_model

    ### Begin actual model training
    # Define some initial parameters, and the early stopping and adaptive learning rate callback
    early_stopping = EarlyStopping(monitor='acc', patience=opt_patience)
    LR_sched = LearningRateScheduler(schedule = adaptive_lr)

    # Potential for adding tensor board functionality to see the change of parameters with time
    #tensor_board = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32,\
    #                            write_graph=True, write_grads=True, write_images=True,\
    #                            embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)

    # Start the timer for the training iterations
    start = time.time()

    # Train the model
    for i in range(num_bunch):
        # Give an update as to how many times we have drawn a new example set
        print('Iteration number:',i+1,'/',num_bunch)

        # Make the examples
        print('Starting training data creation:')
        (x_train, y_train) = ex_create(adr_arr = class_array,
                                       seis_arr = segy_obj.data,
                                       seis_spec = segy_obj,
                                       num_examp = num_examples,
                                       cube_incr = cube_incr,
                                       inp_res = inp_res,
                                       sort_adr = False,
                                       replace_illegals = True)

        print('Finished creating',num_examples,'examples!')

        # Define and reshape the training data
        # x_train = np.expand_dims(x_train,axis=4)

        # Convert labels to one-hot encoding(and if necessary change the data type and scale as needed)
        y_train = keras.utils.to_categorical(y_train, num_classes)

        # See if the user has chosen to implement data_augmentation and implement it if so
        if not data_augmentation:
            print('Not using data augmentation.')
            # Run the model training
            history = model.fit(x=x_train,
                                y=y_train,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[early_stopping, LR_sched],
                                epochs=num_epochs,
                                shuffle=True)

        else:
            # !!! Currently does not work
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                         samplewise_center=False,  # set each sample mean to 0
                                         featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                         samplewise_std_normalization=False,  # divide each input by its std
                                         zca_whitening=False,  # apply ZCA whitening
                                         rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                                         width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                                         height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                                         horizontal_flip=True,  # randomly flip images
                                         vertical_flip=False,    # randomly flip images
                                         shear_range = 0.349, # shear intensity (counter-clockwise direction in radians)
                                         zoom_range = 0.2,   # range for random zoom (float)
                                         rescale = 1.5)  # rescaling factor which multiplies data by the value provided

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            history = model.fit_generator(datagen.flow(x_train,
                                                       y_train,
                                                       batch_size = batch_size),
                                          steps_per_epoch = x_train.shape[0] // batch_size,
                                          epochs = num_epochs,
                                          validation_data = (x_test, y_test))

        # Print the training summary
        print(model.summary())



        # Set the time for one training iteration
        if i == 0:
            end = time.time()
            tot_time = (end-start)*num_bunch



        # Give an update on the time remaining
        rem_time = ((num_bunch-(i+1))/num_bunch)*tot_time

        if rem_time <= 300:
            print('Approximate time remaining of the training:',rem_time,' sec.')
        elif 300 < rem_time <= 60*60:
            minutes = rem_time//60
            seconds = (rem_time%60)*(60/100)
            print('Approximate time remaining of the training:',minutes,' min., ',seconds,' sec.')
        elif 60*60 < rem_time <= 60*60*24:
            hours = rem_time//(60*60)
            minutes = (rem_time%(60*60))*(1/60)*(60/100)
            print('Approximate time remaining of the training:',hours,' hrs., ',minutes,' min., ')
        else:
            days = time_rem//(24*60*60)
            hours = (time_rem%(24*60*60))*(1/60)*((1/60))*(24/100)
            print('Approximate time remaining of the training:',days,' days, ',hours,' hrs., ')


    # Save the trained model if the user has chosen to do so
    if write_out:
        print('Saving model: ...')
        model.save(write_location + '.h5')
        print('Model saved.')


    # Return the trained model
    return model