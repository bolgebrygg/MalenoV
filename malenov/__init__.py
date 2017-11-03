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