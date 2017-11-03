### ---- Functions for visualizing the predictions from the program ----
# Make a plotting function for plotting the features
def plotNNpred(pred,im_per_line,line_num,section):
    # pred: 4D-numpy array with the features in the 4th dimension
    # im_per_line: How many sub plot images to have in each row of the display
    # line_num: what xline to use as a reference
    # section: the section that was used for prediction

    # Define some initial parameters, like the number of features and plot size, etc.
    features = pred.shape[3]
    plt.figure(2, figsize=(20,20))
    n_columns = im_per_line
    n_rows = math.ceil(features / n_columns) + 1

    # Itterate through the sub-plots and fill them with the features, do some simple formatting
    for i in range(features):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Feature ' + str(i+1))
        plt.imshow(pred[:,line_num-1,:,i].T, interpolation="nearest", cmap="rainbow",\
                   extent=[section[0],section[1],-section[5],-section[4]])
        plt.colorbar()


# Make and visualize the predicted data
def visualization(filename,inp_seis,seis_obj,keras_model,cube_incr,section_edge,xline_ref,num_classes,\
                  inp_res=np.float64,sect_form = None,save_pred = False, save_file = 'default_write', \
                  pred_batch = 1,show_feature = False, show_prob = True):
    # filename: filename of the segy-cube to be imported (necessary for copying the segy-frame before writing a new segy)
    # inp_seis: a 3D numpy array that holds the input seismic cube
    # seis_obj: Object returned from the segy_decomp function
    # keras_model: keras model that has been trained previously
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # section_edge: edge locations of the sub-section; either index or (min. inline, max. inline, min. xline, max xline, min z, max z)
    # xline_ref: reference crossline from the original seismic cube to be plotted with the prediction (must be within section)
    # num_classes: number of classes to be predicted
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # sect_form: formatting of the section edges (if 'segy' we have to convert iline,xline,time to indexes)
    # save_pred: whether or not to save the prediction as a segy, numpy and csv(ixz) file.
    # save_file: name of the files to be saved (extensions are added automatically)
    # pred_batch: number of traces to predict on at a time
    # show_features: whether or not to get the features or the classes
    # show_prob: if the user wants to get out probabilities or classifications



    # Adjust the section numbering and reference xline from inline-xline-time to index if this is given in segy format
    if sect_form == 'segy':
        section_edge[0] = (section_edge[0] - seis_obj.inl_start)//seis_obj.inl_step
        section_edge[1] = (section_edge[1] - seis_obj.inl_start)//seis_obj.inl_step
        section_edge[2] = (section_edge[2] - seis_obj.xl_start)//seis_obj.xl_step
        section_edge[3] = (section_edge[3] - seis_obj.xl_start)//seis_obj.xl_step
        section_edge[4] = (section_edge[4] - seis_obj.t_start)//seis_obj.t_step
        section_edge[5] = (section_edge[5] - seis_obj.t_start)//seis_obj.t_step
        xline_ref = (xline_ref - seis_obj.xl_start)//seis_obj.xl_step

    # Make the prediction
    pred = predicting(filename=filename,
                      inp_seis=inp_seis,
                      seis_obj=seis_obj,
                      keras_model=keras_model,
                      cube_incr=cube_incr,
                      num_classes = num_classes,
                      inp_res = inp_res,
                      mode='section',
                      section=section_edge,
                      line_num=0,
                      print_segy=save_pred,
                      savename=save_file,
                      pred_batch = pred_batch,
                      show_features=show_feature,
                      layer_name='attribute_layer',
                      show_prob = show_prob)

    # Define some parameters used for getting nice plots(range of c-axis, and which row to show in the prediction)
    features = pred.shape[2]
    if show_prob:
        class_row = 1
        c_max = 1
    else:
        class_row = 0
        c_max = num_classes-1

    # Visualize the results from the prediction, either with features or classes/probabilities
    if show_feature:
        # Make the figure object/handle and plot the reference xline
        plt.figure(1, figsize=(15,15))
        plt.title('x-line')
        plt.imshow(inp_seis[cube_incr:-cube_incr,xline_ref,cube_incr:-cube_incr].T,interpolation="nearest",\
                   cmap="gray",extent=[cube_incr,-cube_incr+len(inp_seis),cube_incr-len(inp_seis[0,0]),-cube_incr])

        # Plot all the features and show the figures
        plotNNpred(pred,5,xline_ref-section_edge[2]+1,section_edge)
        plt.show()

    else:
        # Make the figure object/handle and plot the reference xline
        plt.figure(1, figsize=(15,15))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        plt.subplot(gs[0])
        plt.title('x-line')
        plt.imshow(inp_seis[cube_incr:-cube_incr,xline_ref,cube_incr:-cube_incr].T,interpolation="nearest",\
                   cmap="gray",extent=[cube_incr,-cube_incr+len(inp_seis),cube_incr-len(inp_seis[0,0]),-cube_incr])
        plt.colorbar()

        # Plot the probability/classification and show the figures
        plt.subplot(gs[1])
        plt.title('classification/probability of 1')
        plt.imshow(pred[:,xline_ref-section_edge[2],:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max),\
                  extent=[section_edge[0],section_edge[1],-section_edge[5],-section_edge[4]])
        plt.colorbar()
        plt.show()

    # Return the predicted numpy cube
    return pred

# Rough function to show more detailed plots of the predictions in python for QC before going to Petrel
def show_details(filename,cube_incr,predic,inline,inl_start,xline,xl_start,\
                 slice_number,slice_incr,inp_format=np.float64,show_prob = True,num_classes = 2):
    # filename: filename of the segy-cube to be imported (necessary for copying the segy-frame before writing a new segy)
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # predic: numpy cube holding the prediction
    # inline: inline number to center our visualization on
    # inl_start: index of the first inline in the prediction
    # xline: xline number to center our visualization on
    # xl_start: index of the first xline in the prediction
    # slice_number: depth slice number to center our visualization on
    # slice_incr: increments to take in depth between each plot
    # inp_format: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # show_prob: if the user wants to get out probabilities or classifications
    # num_classes: number of classes that was predicted


    # Read out the reference segy object
    segy_obj = segy_decomp(segy_file = filename,
                           plot_data = False,
                           read_direc = 'xline',
                           inp_res = inp_format)

    # Get the numpy cube from the reference segy object
    inp_seis = segy_obj.data

    # define some parameters used for getting nice plots(range of c-axis, and which row to show in the prediction)
    if show_prob:
        class_row = 1
        c_max = 1
    else:
        class_row = 0
        c_max = num_classes-1

    # Make the figure object/handle and plot the reference xline
    plt.figure(1, figsize=(20,15))
    plt.subplot(1, 8, 1)
    plt.title('xline: ' + str(xline))
    plt.imshow(inp_seis[inline-cube_incr:inline+cube_incr,xline,cube_incr:-cube_incr].T,interpolation="nearest", cmap="gray")
    plt.colorbar()

    # Plot the prediciton for the reference xline along with 3 increments in each direction
    plt.subplot(1, 8, 2)
    plt.title('xline - 3')
    plt.imshow(predic[:,xline-xl_start - 3,:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))

    plt.subplot(1, 8, 3)
    plt.title('xline')
    plt.imshow(predic[:,xline-xl_start,:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))

    plt.subplot(1, 8, 4)
    plt.title('xline + 3')
    plt.imshow(predic[:,xline-xl_start + 3,:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))
    plt.colorbar()

    # Plot the reference inline
    plt.subplot(1, 8, 1+4)
    plt.title('inline: ' + str(inline))
    plt.imshow(inp_seis[inline,xline-cube_incr:xline+cube_incr,cube_incr:-cube_incr].T,interpolation="nearest", cmap="gray")
    plt.colorbar()

    # Plot the prediciton for the reference inline along with 3 increments in each direction
    plt.subplot(1, 8, 2+4)
    plt.title('inline - 3')
    plt.imshow(predic[inline-inl_start-3,:,:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))

    plt.subplot(1, 8, 3+4)
    plt.title('inline')
    plt.imshow(predic[inline-inl_start,:,:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))

    plt.subplot(1, 8, 4+4)
    plt.title('inline + 3')
    plt.imshow(predic[inline-inl_start+3,:,:,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))
    plt.colorbar()

    # Make a new figure object/handle and plot 3 reference depth slices
    plt.figure(2, figsize=(20,5))
    plt.subplot(1, 3, 1)
    plt.title('slice - ' + str(slice_incr))
    plt.imshow(inp_seis[inline-cube_incr:inline+cube_incr,xline-cube_incr:xline+cube_incr,cube_incr+slice_number-slice_incr].T,\
               interpolation="nearest", cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title('slice: ' + str(slice_number))
    plt.imshow(inp_seis[inline-cube_incr:inline+cube_incr,xline-cube_incr:xline+cube_incr,cube_incr+slice_number].T,\
               interpolation="nearest", cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title('slice + ' + str(slice_incr))
    plt.imshow(inp_seis[inline-cube_incr:inline+cube_incr,xline-cube_incr:xline+cube_incr,cube_incr+slice_number+slice_incr].T,\
               interpolation="nearest", cmap="gray")
    plt.colorbar()

    # Make a new figure object/handle and plot the 3 corresponding predicted depth slices
    plt.figure(3, figsize=(20,5))
    plt.subplot(1, 3, 1)
    plt.title('slice - ' + str(slice_incr))
    plt.imshow(predic[:,:,slice_number-slice_incr,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))

    plt.subplot(1, 3, 2)
    plt.title('slice: ' + str(slice_number))
    plt.imshow(predic[:,:,slice_number,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))

    plt.subplot(1, 3, 3)
    plt.title('slice + ' + str(slice_incr))
    plt.imshow(predic[:,:,slice_number+slice_incr,class_row].T,interpolation="nearest", cmap="gist_rainbow", clim=(0.0, c_max))
    plt.colorbar()
    plt.show()