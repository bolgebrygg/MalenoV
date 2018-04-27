# Make initial package imports
import numpy as np
import random
import keras
import time

from MalenoV_func.modelling import *
from MalenoV_func.data_aug import *

from keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard

### ---- Functions for the training part of the program ----
# Make a function that combines the adress cubes and makes a list of class adresses
def convert(file_list, save = False, savename = 'adress_list', ex_adjust = False):
    # file_list: list of file names(strings) of adresses for the different classes
    # save: boolean that determines if a new ixz file should be saved with adresses and class numbers
    # savename: desired name of new .ixz-file
    # ex_adjust: boolean that determines if the amount of each class should be approximately equalized

    # Make an array of that holds the number of each example provided, if equalization is needed
    if ex_adjust:
        len_array = np.zeros(len(file_list),dtype = np.float32)
        for i in range(len(file_list)):
            len_array[i] = len(np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.float32))

        # Cnvert this array to a multiplier that determines how many times a given class set needs to be
        len_array /= max(len_array)
        multiplier = 1//len_array


    # preallocate space for the adr_list, the output containing all the adresses and classes
    adr_list = np.empty([0,4], dtype = np.int32)

    # Itterate through the list of example adresses and store the class as an integer
    for i in range(len(file_list)):
        a = np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.int32)
        adr_list = np.append(adr_list,np.append(a,i*np.ones((len(a),1),dtype = np.int32),axis=1),axis=0)

        # If desired copy the entire list by the multiplier calculated
        if ex_adjust:
            for k in range(int(multiplier[i])-1):
                adr_list = np.append(adr_list,np.append(a,i*np.ones((len(a),1),dtype = np.int32),axis=1),axis=0)

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename + '.ixz', adr_list, fmt = '%i')

    # Return the list of adresses and classes as a numpy array
    return adr_list



# Function for example creating
# Outputs a dictionary with pairs of cube tuples and labels
def ex_create(adr_arr,seis_arr,seis_spec,num_examp,cube_incr,inp_res=np.float64,sort_adr = False,replace_illegals = True):
    # adr_arr: 4-column numpy matrix that holds a header in the first row, then adress and class information for examples
    # seis_arr: 3D numpy array that holds a seismic cube
    # seis_spec: object that holds the specifications of the seismic cube;
    # num_examp: the number of output mini-cubes that should be created
    # cube_incr: the number of increments included in each direction from the example to make a mini-cube
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # sort_adr: boolean; whether or not to sort the randomly drawn adresses before making the example cubes
    # replace_illegals: boolean; whether or not to draw a new sample in place for an illegal one, or not

    # Define the cube size
    cube_size = 2*cube_incr+1

    # Define some boundary parameters given in the input object
    inline_start = seis_spec.inl_start
    inline_end = seis_spec.inl_end
    inline_step = seis_spec.inl_step
    xline_start = seis_spec.xl_start
    xline_end = seis_spec.xl_end
    xline_step = seis_spec.xl_step
    t_start = seis_spec.t_start
    t_end = seis_spec.t_end
    t_step = seis_spec.t_step
    num_channels = seis_spec.cube_num

    # Define the buffer zone around the edge of the cube that defines the legal/illegal adresses
    inl_min = inline_start + inline_step*cube_incr
    inl_max = inline_end - inline_step*cube_incr
    xl_min = xline_start + xline_step*cube_incr
    xl_max = xline_end - xline_step*cube_incr
    t_min = t_start + t_step*cube_incr
    t_max = t_end - t_step*cube_incr

    # Print the buffer zone edges
    print('Defining the buffer zone:')
    print('(inl_min,','inl_max,','xl_min,','xl_max,','t_min,','t_max)')
    print('(',inl_min,',',inl_max,',',xl_min,',',xl_max,',',t_min,',',t_max,')')
    # Also give the buffer values in terms of indexes
    print('(',cube_incr,',',((inline_end-inline_start)//inline_step) - cube_incr,\
          ',',cube_incr,',',((xline_end-xline_start)//xline_step) - cube_incr,\
          ',',cube_incr,',',((t_end-t_start)//t_step) - cube_incr,')')

    # We preallocate the function outputs; a list of examples and a list of labels
    examples = np.empty((num_examp,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
    labels = np.empty(num_examp,dtype=np.int8)

    # If we want to stack the examples in the third dimension we use the following example preallocation in stead
    # examples = np.empty((num_examp*(cube_size),(cube_size),(cube_size)),dtype=inp_res)

    # Generate a random list of indexes to be drawn, and make sure it only takes a legal amount of examples
    try:
        max_row_idx = len(adr_arr)-1
        rand_idx = random.sample(range(0, max_row_idx), num_examp)
        # NOTE: Could be faster to sort indexes before making examples for algorithm optimization
        if sort_adr:
            rand_idx.sort()
    except ValueError:
        print('Sample size exceeded population size.')

    # Make an iterator for when the lists should become shorter(if we have replacement of illegals or not)
    n=0
    for i in range(num_examp):
        # Get a random in-line, x-line, and time value, and store the label
        # Make sure there is room for an example at this index
        for j in range(50):
            adr = adr_arr[rand_idx[i]]
            # Check that the given example is within the legal zone
            if (adr[0]>=inl_min and adr[0]<inl_max) and \
                (adr[1]>=xl_min and adr[1]<xl_max) and \
                (adr[2]>=t_min and adr[2]<t_max):
                # Make the example for the given address
                # Convert the adresses to indexes and store the examples in the 4th dimension
                idx = [(adr[0]-inline_start)//inline_step,(adr[1]-xline_start)//xline_step,(adr[2]-t_start)//t_step]


                examples[i-n,:,:,:,:] = seis_arr[idx[0]-cube_incr:idx[0]+cube_incr+1,\
                              idx[1]-cube_incr:idx[1]+cube_incr+1,\
                              idx[2]-cube_incr:idx[2]+cube_incr+1,:]

                # Put the cube and label into the lists
                labels[i-n] = adr[-1]

                # Alternatively; stack the examples in the third dimension
                #datasets[(i-n)*(cube_size):(i-n+1)*(cube_size),:,:] = ex
                break
            else:
                # If we want to replace the illegals, draw again
                if replace_illegals:
                    rand_idx[i] = random.randint(0,max_row_idx)
                else:
                    # if not, just make the output lists shorter
                    n += 1
                    break

            if j == 50:
                # If we can't get a proper cube in 50 consequtive tries
                print('Badly conditioned dataset!')

    # Slice the data if desired
    #labels = labels[0:i-n+1]
    #examples = examples[0:i-n+1,:,:,:]

    # Return the output list/tuple (slice it if it has been shortened)
    return (examples[0:i-n+1,:,:,:,:], labels[0:i-n+1])


# Function that takes the epoch as input and returns the desired learning rate
def adaptive_lr(input_int):
    # input_int: the epoch that is currently being entered

    # define the learning rate (quite arbitrarily decaying)
    lr = 0.1**input_int

    #return the learning rate
    return lr


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

        model = make_model(cube_size = cube_size,
                           num_channels = num_channels,
                           num_classes = num_classes)


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
