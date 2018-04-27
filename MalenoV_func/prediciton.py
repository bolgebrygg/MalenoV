# Make initial package imports
import numpy as np
import time
import segyio

from keras.models import Sequential, Model
from shutil import copyfile

### ---- Functions for the prediction part of the program ----
# Parse the cube into sub-cubes suitable as model input
def cube_parse(seis_arr,cube_incr,inp_res = np.float64, mode = 'trace', padding = False,\
               conc = False, inline_num = 0, xline_num = 0, depth = 0):
    # seis_arr: a 3D numpy array that holds a seismic cube
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # mode: how much of the 3D-cube should be converted to examples ('full','xline','inline','trace', or 'point')
    # padding: do we want to pad the zone which is outside our buffer with zeroes?
    # conc: do we want to concattenate the examples, or store them in the same matrix they were fed to us?
    # inline_num: if mode is inline or point; what inline do we use?
    # xline_num: if mode is xline or point; what xline do we use?
    # depth: if mode is point; what depth do we use?

    # Make some initial definitions wrt. dimensionality
    inls = seis_arr.shape[0]
    xls = seis_arr.shape[1]
    zls = seis_arr.shape[2]
    num_channels = seis_arr.shape[3]
    cube_size = 2*cube_incr+1

    # Define the indent where the saved data will start, if user wants padding this is 0, else it is cube_incr
    if padding:
        i_re = 0
        x_re = 0
        z_re = 0
        # Preallocate the output array, if concatenated it's 4 dimensional, if not it's 6 dimensional
        if conc:
            # Make adjustments to the parameters so that we iterate over the right number of samples, etc.
            if mode == 'full':
                examples = np.zeros((inls*xls*zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
            elif mode == 'inline':
                examples = np.zeros((xls*zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                x_re = cube_incr
            elif mode == 'xline':
                examples = np.zeros((inls*zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                i_re = cube_incr
            elif mode == 'trace':
                examples = np.zeros((zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
            elif mode == 'point':
                examples = np.zeros((1,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
                z_re = cube_incr
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
            # Take into account that we will have a total smaller dimensionality of data due to illegals
            inls -= 2*cube_incr
            xls -= 2*cube_incr
            zls -= 2*cube_incr
        else:
            # Make adjustments to the parameters so that we iterate over the right number of samples, etc.
            if mode == 'full':
                examples = np.zeros((inls,xls,zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
            elif mode == 'inline':
                examples = np.zeros((1,xls,zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                x_re = cube_incr
            elif mode == 'xline':
                examples = np.zeros((inls,1,zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                i_re = cube_incr
            elif mode == 'trace':
                examples = np.zeros((1,1,zls,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
            elif mode == 'point':
                examples = np.zeros((1,1,1,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                i_re = cube_incr
                x_re = cube_incr
                z_re = cube_incr
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
    else:
        i_re = cube_incr
        x_re = cube_incr
        z_re = cube_incr
        # Preallocate the output array, if concatenated it's 5 dimensional, if not it's 7 dimensional
        if conc:
            # Make adjustments to the parameters so that we iterate over the right number of samples, etc.
            if mode == 'full':
                examples = np.empty(((inls-2*cube_incr)*(xls-2*cube_incr)*(zls-2*cube_incr),cube_size,cube_size,cube_size,num_channels),\
                                   dtype=inp_res)
            elif mode == 'inline':
                examples = np.empty(((xls-2*cube_incr)*(zls-2*cube_incr),cube_size,cube_size,cube_size,num_channels),\
                                   dtype=inp_res)
                inline_num -= cube_incr
                xline_num = 0
                depth = 0
            elif mode == 'xline':
                examples = np.empty(((inls-2*cube_incr)*(zls-2*cube_incr),cube_size1,cube_size,cube_size,num_channels),\
                                   dtype=inp_res)
                inline_num = 0
                xline_num -= cube_incr
                depth = 0
            elif mode == 'trace':
                examples = np.empty((zls-2*cube_incr,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                inline_num -= cube_incr
                xline_num -= cube_incr
                depth = 0
            elif mode == 'point':
                examples = np.empty((1,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
                inline_num -= cube_incr
                xline_num -= cube_incr
                depth -= cube_incr
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')
            # Take into account that we will have a total smaller dimensionality of data due to illegals
            inls -= 2*cube_incr
            xls -= 2*cube_incr
            zls -= 2*cube_incr
        else:
            if mode == 'full':
                examples = np.empty(((inls-2*cube_incr),(xls-2*cube_incr),(zls-2*cube_incr),cube_size,cube_size,cube_size,num_channels),\
                                   dtype=inp_res)
            elif mode == 'inline':
                examples = np.empty((1,(xls-2*cube_incr),(zls-2*cube_incr),cube_size,cube_size,cube_size,num_channels),\
                                   dtype=inp_res)
            elif mode == 'xline':
                examples = np.empty(((inls-2*cube_incr),1,(zls-2*cube_incr),cube_size,cube_size,cube_size,num_channels),\
                                   dtype=inp_res)
            elif mode == 'trace':
                examples = np.empty((1,1,(zls-2*cube_incr),cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
            elif mode == 'point':
                examples = np.empty((1,1,1,cube_size,cube_size,cube_size,num_channels),dtype=inp_res)
            else:
                print('ERROR: invalid mode! use: ''full'',''xline'',''inline'',''trace'', or ''point''')


    # Iterate through the desired section of the 3D input array, create the example cubes, and store them as desired
    if conc:
        # Make the cubes
        for i in range(cube_incr, inls+cube_incr):
            if mode == 'xline':
                j = xline_num
                for k in range(cube_size, zls+cube_size):
                    examples[inls*(i-i_re)+k-z_re,:,:,:,:] = seis_arr[i-cube_incr+inline_num:i+cube_incr+inline_num+1,\
                                                                    j-cube_incr:j+cube_incr+1,\
                                                                    k-cube_incr+depth:k+cube_incr+depth+1,:]
            else:
                for j in range(cube_incr, xls+cube_incr):
                    for k in range(cube_incr, zls+cube_incr):
                        examples[(i-i_re)*inls+(j-x_re)*xls+k-z_re,:,:,:,:] = \
                                                            seis_arr[i-cube_incr+inline_num:i+cube_incr+inline_num+1,\
                                                                     j-cube_incr+xline_num:j+cube_incr+xline_num+1,\
                                                                     k-cube_incr+depth:k+cube_incr+depth+1,:]

                        # Make sure we stop after the appropriate number of iterations
                        if mode == 'point':
                            break
                    if mode == 'point' or mode == 'trace':
                        break
                if mode == 'point' or mode == 'trace' or mode == 'inline':
                    break


    else:
        # Make the cubes
        for i in range(cube_incr, inls-cube_incr):
            if mode == 'xline':
                for k in range(cube_incr, zls-cube_incr):
                    examples[i-i_re,1,k-z_re,:,:,:,:] = seis_arr[i-cube_incr:i+cube_incr+1,\
                                                               xline_num-cube_incr:xline_num+cube_incr+1,\
                                                               k-cube_incr:k+cube_incr+1,:]
            else:
                for j in range(cube_incr, xls-cube_incr):
                    for k in range(cube_incr, zls-cube_incr):
                        examples[i-i_re,j-x_re,k-z_re,:,:,:,:] = seis_arr[i+inline_num-cube_incr:i+inline_num+cube_incr+1,\
                                                                        j+xline_num-cube_incr:j+xline_num+cube_incr+1,\
                                                                        k+depth-cube_incr:k+depth+cube_incr+1,:]

                        # Make sure we stop after the appropriate number of iterations
                        if mode == 'point':
                            break
                    if mode == 'point' or mode == 'trace':
                        break
                if mode == 'point' or mode == 'trace' or mode == 'inline':
                    break


    # Return the list of examples stored as the desired type of array
    return examples



# Make an intermediate output model to check filters
def makeIntermediate(keras_model,layer_name):
    # keras_model: keras model that has been trained previously
    # layer_name: name of the layer with the desired output

    # Define the new model that stops at the desired layer
    intermediate_layer_model = Model(inputs=keras_model.input,\
                                     outputs=keras_model.get_layer(layer_name).output)

    # Return the newly defined model
    return intermediate_layer_model



# Predict the output class of the given input traces
def predicting(filename,inp_seis,seis_obj,keras_model,cube_incr,num_classes,inp_res=np.float64,mode='xline',\
               section=np.asarray([0,0,0,0,0,0]),line_num=0, print_segy = False,savename = 'default_write',\
               pred_batch = 1,show_features = False, layer_name='attribute_layer', show_prob = False):
    # filename: filename of the segy-cube to be imported (necessary for copying the segy-frame before writing a new segy)
    # inp_seis: a 3D numpy array that holds the input seismic cube
    # seis_obj: Object returned from the segy_decomp function
    # keras_model: keras model that has been trained previously
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # num_classes: num_classes: number of destinct classes we are training on
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # mode: what part of the cube to predict along; 'inline', 'xline', 'section, or 'full' (entire cube)
    # section: edge locations(index) of the sub-section (min. inline, max. inline, min. xline, max xline, min z, max z)
    # line_num: xline/inline number to predict along
    # print_segy: whether or not to save the prediction as a segy, npy and csv file (previously just segy)
    # savename: name of the files to be saved (extensions are added automatically)
    # pred_batch: number of traces to predict on at a time
    # show_features: whether or not to get the features or the classes
    # layer_name: optionally give a different layer to get the features from (name defined in keras.model)
    # show_prob: if the user wants to get out probabilities or classifications

    # Define some initial parameters
    num_channels = seis_obj.cube_num
    inls = inp_seis.shape[0]
    xls = inp_seis.shape[1]
    zls = inp_seis.shape[2]
    cube_size = 2*cube_incr+1

    # If the user simply wants the classification we only need 1 value for each input point,
    if not show_prob:
        num_classes = 1

    # Read the section needed for prediction depending on the mode
    if mode == 'inline':
        section_edge = np.asarray([line_num,line_num,cube_incr,xls-cube_incr,cube_incr,zls-cube_incr])
    elif mode == 'xline':
        section_edge = np.asarray([cube_incr,inls-cube_incr,line_num,line_num,cube_incr,zls-cube_incr])
    elif mode == 'section':
        section_edge = section
    elif mode == 'full':
        section_edge = np.asarray([cube_incr,inls-cube_incr,cube_incr,xls-cube_incr,cube_incr,zls-cube_incr])
    else:
        print('invalid mode, please input inline, xline, section, or full')

    # Preallocate the full prediction array and if the user wants to show the features make the intermediate model,
    if show_features:
        intermediate_layer_model = Model(inputs=keras_model.input,
                                         outputs=keras_model.get_layer(layer_name).output)
        prediction = np.empty((\
            (section_edge[5]-section_edge[4]+1)*(section_edge[3]-section_edge[2]+1)*(section_edge[1]-section_edge[0]+1),10),\
                              dtype=np.float32)
    else:
        prediction = np.empty((\
            (section_edge[5]-section_edge[4]+1)*(section_edge[3]-section_edge[2]+1)*(section_edge[1]-section_edge[0]+1),\
                               num_classes),dtype=np.float32)

    # Preallocate the data array to fill for each batch and initiate iterators
    data = np.empty((pred_batch*(section_edge[5]-section_edge[4]+1),cube_size,cube_size,cube_size,num_channels), dtype=inp_res)
    indx = 0
    jndx = 0

    # Calculate how many sets of batches need to be done and define parameters needed for the final batch
    tot_len = (section[1]-section[0]+1)*(section[3]-section[2]+1)
    rem = tot_len % pred_batch
    num_it = tot_len // pred_batch
    # Time the sub_prediction
    start = time.time()

    # Start making sub-cubes from the input traces and store then in the data array
    print('Retrieving to memory:')
    for il_num in range(section_edge[0],section_edge[1]+1):
        # Make a progres update for the inline number
        print('inline-num:',il_num-section_edge[0]+1,'/',section_edge[1]-section_edge[0]+1)
        for xl_num in range(section_edge[2],section_edge[3]+1):
            # Make a progres update for the xline number
            print('xline-num:',xl_num-section_edge[2]+1,'/',section_edge[3]-section_edge[2]+1)
            for z_num in range(section_edge[5]-section_edge[4]+1):
                # Call the cube_parse function to get the cubes corresponding to the current point
                data[indx*(section_edge[5]-section_edge[4]+1)+z_num,:,:,:,:] = cube_parse(seis_arr = inp_seis,
                                                                                        cube_incr = cube_incr,
                                                                                        inp_res = inp_res,
                                                                                        mode = 'point',
                                                                                        padding = False,
                                                                                        conc = True,
                                                                                        inline_num = il_num,
                                                                                        xline_num = xl_num,
                                                                                        depth = z_num+section_edge[4])

            # Check if we have filled up the data array and need to do a prediction
            if (indx+1) % pred_batch == 0:
                print('Making prediction on sub-section:')

                # Predict the given class or features dependant on the user input
                if show_features:
                    prediction[jndx*(pred_batch*(section_edge[5]-section_edge[4]+1)):\
                              (jndx+1)*(pred_batch*(section_edge[5]-section_edge[4]+1)),:] = \
                                    intermediate_layer_model.predict((data))

                else:
                    if show_prob:
                        # Simple model prediction with probabilities
                        prediction[jndx*(pred_batch*(section_edge[5]-section_edge[4]+1)):\
                                   (jndx+1)*(pred_batch*(section_edge[5]-section_edge[4]+1)),:] = \
                        keras_model.predict((data))
                    else:
                        # Model prediction of classes
                        prediction[jndx*(pred_batch*(section_edge[5]-section_edge[4]+1)):\
                                   (jndx+1)*(pred_batch*(section_edge[5]-section_edge[4]+1)),:] = \
                        np.expand_dims(keras_model.predict_classes((data)),axis = 1)

                # Tell the user the section is finished
                print('Section finished!')

                if jndx == 0:
                    # Finish the timer and calculate how long the user should expect the program to take:
                    end = time.time()
                    DT = end-start # seconds per iteration
                    tot_time = num_it*DT+(rem/pred_batch)*DT #seconds

                # Give the user an update regarding the time remaining
                time_rem = (tot_time-DT*(jndx+1))
                if time_rem <= 300:
                    print('Approximate time remaining of the prediction:',time_rem, ' sec.')
                elif 300 < time_rem <= 60*60:
                    minutes = time_rem//60
                    seconds = (time_rem%60)*(60/100)
                    print('Approximate time remaining of the prediction:',minutes,' min., ',seconds,' sec.')
                elif 60*60 < time_rem <= 60*60*24:
                    hours = time_rem//(60*60)
                    minutes = (time_rem%(60*60))*(1/60)*(60/100)
                    print('Approximate time remaining of the prediction:',hours,' hrs., ',minutes,' min., ')
                else:
                    days = time_rem//(24*60*60)
                    hours = (time_rem%(24*60*60))*(1/60)*((1/60))*(24/100)
                    print('Approximate time remaining of the prediction:',days,' days, ',hours,' hrs., ')


                # Update iterators and give updates to user
                indx = 0
                jndx+=1
                print('Retrieving to memory:')

            # Check if we have exhausted the range of data to be predicted and need to finish the function
            elif jndx == num_it and indx == rem-1:
                # Slice the data array to only include the relevant part
                data = data[:indx*(section_edge[5]-section_edge[4]+1)+z_num+1]

                print('Finalizing prediction:')

                # Make the final prediction
                if show_features:
                    prediction[jndx*(pred_batch*(section_edge[5]-section_edge[4]+1)):,:] = \
                                    intermediate_layer_model.predict((data))
                else:
                    if show_prob:
                        prediction[jndx*(pred_batch*(section_edge[5]-section_edge[4]+1)):,:] = \
                                    keras_model.predict((data))
                    else:
                        prediction[jndx*(pred_batch*(section_edge[5]-section_edge[4]+1)):,:] = \
                                    np.expand_dims(keras_model.predict_classes((data)),axis = 1)

            # If we should keep filling the data and not predict yet, simply increase the iterator
            else:
                indx+=1

    # Reshape the prediction to the shape of the desired cube
    print('Reshaping prediction:')
    if show_features:
        prediction = prediction.reshape((section_edge[1]-section_edge[0]+1,\
                                         section_edge[3]-section_edge[2]+1,\
                                         section_edge[5]-section_edge[4]+1,10),order='C')
    else:
        prediction = prediction.reshape((section_edge[1]-section_edge[0]+1,\
                                         section_edge[3]-section_edge[2]+1,\
                                         section_edge[5]-section_edge[4]+1,num_classes),order='C')

    print('Prediction finished!')

    # Save the prediction as a segy, numpy and csv file
    # NOTE: Everything SEGY and CSV is made into 32bit-float to conform to commonly used reading programs
    if print_segy:
        # Update the data we send to the saver functions dependant on what we have predicted
        if show_prob:
            class_row = 1
        else:
            class_row = 0

        print('Saving prediction: ...')

        # Save the numpy file
        np.save(savename + '.npy', prediction)

        # Get the right filename in case the input is given as a list
        if type(filename) is list:
            # Save the segy file using the input filename as a framework
            # Just use the first member of the list as the reference
            input_file = filename[0]
        else:
            # Save the segy file using the input filename as a framework
            input_file=filename

        output_file=savename + '.sgy'

        copyfile(input_file, output_file)

        with segyio.open( output_file, "r+" ) as src:
            # iterate through each inline and update the values
            i = 0
            for ilno in src.ilines:
                src.iline[ilno] = -1*(np.ones((src.iline[ilno].shape),dtype = np.float32))

                if src.ilines[section_edge[0]] <= ilno <= src.ilines[section_edge[1]]:
                    line = src.iline[ilno]
                    line[section_edge[2]:section_edge[3]+1,section_edge[4]:section_edge[5]+1] = prediction[i,:,:,class_row]
                    src.iline[ilno]=line
                    i += 1

        # Save the csv(ixz) file
        csv_struct(inp_numpy = prediction[:,:,:,class_row],
                   spec_obj = seis_obj,
                   section = section_edge,
                   inp_res = np.float32,
                   save = True,
                   savename = (savename + '.ixz'))

        # Print to the user that the function has finished saving
        print('Prediction saved.')

    # Return the prediction array
    return prediction
