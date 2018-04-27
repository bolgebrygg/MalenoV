### Function for n-dimensional seismic facies training /classification using Convolutional Neural Nets (CNN)
### By: Charles Rutherford Ildstad (University of Trondheim), as part of a summer intern project in ConocoPhillips and private work
### Contributions from Anders U. Waldeland (University of Oslo), Chris Olsen (ConocoPhillips), Doug Hakkarinen (ConocoPhillips)
### Date: 26.10.2017
### For: ConocoPhillips, Norway,
### GNU V3.0 lesser license

# Make initial package imports
import segyio
import random
import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv3D
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from matplotlib import gridspec

from keras.layers.normalization import BatchNormalization

from shutil import copyfile

## Import the needed files
from MalenoV_func.data_aug import *
from MalenoV_func.masterf import *
from MalenoV_func.make_model import *
from MalenoV_func.prediction import *
from MalenoV_func.segy_files import *
from MalenoV_func.training import *
from MalenoV_func.visualize import *


# Set random seed for reproducability
np.random.seed(7)
# Confirm backend if in doubt
#keras.backend.backend()


#### ---- Run an instance of the master function ----
filenames=['PGS16902VIK_pstm_ang08_17_dec_crop','PGS16902VIK_pstm_ang17_27_dec_crop', 'PGS16902VIK_pstm_ang27_33_dec_crop']    # name of the segy-cube(s) with data
inp_res = np.float32    # formatting of the input seismic (e.g. np.int8 for 8-bit data, np.float32 for 32-bit data, etc)
cube_incr = 32    # number of increments in each direction to create a training cube

# Define the dictionary holding all the training parameters
train_dict = {
    'files' : ['multi_channel_new.pts','multi_coherent.pts','multi_else.pts','multi_fault.pts','multi_grid.pts','multi_grizzly.pts'],    # list of names of class-adresses
    'num_tot_iterations': 25,    # number of times we draw a new training ensemble/mini-batch
    'epochs' : 12,    # number of epochs we run on each training ensemble/mini-batch
    'num_train_ex' : 18000,    # number of training examples in each training ensemble/mini-batch
    'batch_size' : 32,    # number of training examples fed to the optimizer as a batch
    'opt_patience' : 10,    # number of epochs with the same accuracy before force breaking the training ensemble/mini-batch
    'data_augmentation' : False,    # whether or not we are using data augmentation
    'save_model' : True,    # whether or not we are saving the trained model
    'save_location' : 'F3_train'    # file name for the saved trained model
}

# Define the dictionary holding all the prediction parameters
pred_dict = {
    'keras_model' :  keras.models.load_model('mulitvolume_multifacies.h5'), # input model to be used for prediction, to load a model use: keras.models.load_model('write_location')
    'section_edge' : np.asarray([33282, 33282, 123898, 123900, 128, 2840]), # inline and xline section to be predicted (all depths), must contain xline
    'show_feature' : False,    # Show the distinct features before they are combined to a prediction
    'xline' : 123900,    # xline used for classification (index)(should be within section range)
    'num_class' : len(train_dict['files']),    # number of classes to output
    'cord_syst' : 'segy',    # Coordinate system used, default is 0,0. Set to 'segy' to give inputs in (inline,xline)
    'save_pred' : True,    # Save the prediction as a segy-cube
    'save_location' : 'sunday.segy',     # file name for the saved prediction
    'pred_batch' : 25,     # number of traces used to make batches of mini-cubes that are stored in memory at once
    #'pred_batch' : train_dict['num_train_ex']//(pred_dict['section_edge'][5]-pred_dict['section_edge'][4])    #Suggested value
    'pred_prob' : False     # Give the probabilities of the first class(True), or simply show where each class is classified(False)
}


# Run the master function and save the output in the output dictionary output_dict
output_dict1 = master(
    segy_filename = filenames,     # Seismic filenames
    inp_format = inp_res,     # Format of input seismic
    cube_incr = cube_incr,     # Increments in each direction to create a training cube
    train_dict = train_dict,     # Input training dictionary
    pred_dict = pred_dict,     # Input prediction dictionary
    mode = 'predict'     # Input mode ('train', 'predict', or 'full' for both training AND prediction)
)


# Show additional details about the prediciton
#show_details(
#    filename,
#    cube_incr,
#    output_dict['pred'],
#    inline = 100,
#    inl_start = 75,
#    xline = 169,
#    xl_start = 155,
#    slice_number = 400,
#    slice_incr = 3
#)



### Save/load functions
## returns a prediction cube
## identical to the one saved
#prediction = np.load('filename.npy')
#
## returns a compiled model
## identical to the one saved
#loaded_model = keras.models.load_model('filename.h5')
