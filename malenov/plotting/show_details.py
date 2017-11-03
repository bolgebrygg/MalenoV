# Rough function to show more detailed plots of the predictions in python for QC before going to Petrel
import numpy as np
import matplotlib.pyplot as plt

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