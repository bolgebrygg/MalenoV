# Make and visualize the predicted data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from malenov.plotting import plotNNpred
from ..predict import predicting


def visualization(filename, inp_seis, seis_obj, keras_model, cube_incr, section_edge, xline_ref, num_classes,
                  inp_res=np.float64, sect_form=None, save_pred=False, save_file='default_write',
                  pred_batch=1, show_feature=False, show_prob=True):
    # filename:      filename of the segy-cube to be imported (for copying the segy-frame before writing)
    # inp_seis:      a 3D numpy array that holds the input seismic cube
    # seis_obj:      Object returned from the segy_decomp function
    # keras_model:   keras model that has been trained previously
    # cube_incr:     number of increments included in each direction from the example to make a mini-cube
    # section_edge:  edge locations of the sub-section; either index or
    #                (min. inline, max. inline, min. xline, max xline, min z, max z)
    # xline_ref:     reference crossline from the original seismic cube
    #                to be plotted with the prediction (must be within section)
    # num_classes:   number of classes to be predicted
    # inp_res:       input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # sect_form:     formatting of the section edges (if 'segy' we have to convert iline,xline,time to indexes)
    # save_pred:     whether or not to save the prediction as a segy, numpy and csv(ixz) file.
    # save_file:     name of the files to be saved (extensions are added automatically)
    # pred_batch:    number of traces to predict on at a time
    # show_features: whether or not to get the features or the classes
    # show_prob:     if the user wants to get out probabilities or classifications

    # Adjust the section numbering and reference xline from inline-xline-time to index if this is given in segy format
    if sect_form == 'segy':
        section_edge[0] = (section_edge[0] - seis_obj.inl_start) // seis_obj.inl_step
        section_edge[1] = (section_edge[1] - seis_obj.inl_start) // seis_obj.inl_step
        section_edge[2] = (section_edge[2] - seis_obj.xl_start) // seis_obj.xl_step
        section_edge[3] = (section_edge[3] - seis_obj.xl_start) // seis_obj.xl_step
        section_edge[4] = (section_edge[4] - seis_obj.t_start) // seis_obj.t_step
        section_edge[5] = (section_edge[5] - seis_obj.t_start) // seis_obj.t_step
        xline_ref = (xline_ref - seis_obj.xl_start) // seis_obj.xl_step

    # Make the prediction
    pred = predicting(filename=filename,
                      inp_seis=inp_seis,
                      seis_obj=seis_obj,
                      keras_model=keras_model,
                      cube_incr=cube_incr,
                      num_classes=num_classes,
                      inp_res=inp_res,
                      mode='section',
                      section=section_edge,
                      line_num=0,
                      print_segy=save_pred,
                      savename=save_file,
                      pred_batch=pred_batch,
                      show_features=show_feature,
                      layer_name='attribute_layer',
                      show_prob=show_prob)

    # Define some parameters used for getting nice plots(range of c-axis, and which row to show in the prediction)
    features = pred.shape[2]
    if show_prob:
        class_row = 1
        c_max = 1
    else:
        class_row = 0
        c_max = num_classes - 1

    # Visualize the results from the prediction, either with features or classes/probabilities
    if show_feature:
        # Make the figure object/handle and plot the reference xline
        plt.figure(1, figsize=(15, 15))
        plt.title('x-line')
        plt.imshow(inp_seis[cube_incr:-cube_incr, xline_ref, cube_incr:-cube_incr].T, interpolation="nearest",
                   cmap="gray",
                   extent=[cube_incr, -cube_incr + len(inp_seis), cube_incr - len(inp_seis[0, 0]), -cube_incr])

        # Plot all the features and show the figures
        plotNNpred(pred, 5, xline_ref - section_edge[2] + 1, section_edge)
        plt.show()

    else:
        # Make the figure object/handle and plot the reference xline
        plt.figure(1, figsize=(15, 15))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        plt.subplot(gs[0])
        plt.title('x-line')
        plt.imshow(inp_seis[cube_incr:-cube_incr, xline_ref, cube_incr:-cube_incr].T, interpolation="nearest",
                   cmap="gray",
                   extent=[cube_incr, -cube_incr + len(inp_seis), cube_incr - len(inp_seis[0, 0]), -cube_incr])
        plt.colorbar()

        # Plot the probability/classification and show the figures
        plt.subplot(gs[1])
        plt.title('classification/probability of 1')
        plt.imshow(pred[:, xline_ref - section_edge[2], :, class_row].T, interpolation="nearest", cmap="gist_rainbow",
                   clim=(0.0, c_max),
                   extent=[section_edge[0], section_edge[1], -section_edge[5], -section_edge[4]])
        plt.colorbar()
        plt.show()

    # Return the predicted numpy cube
    return pred
