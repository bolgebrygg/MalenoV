# Stretching
import tensorflow as tf
def randomStretch(window_function, strech):
    return tf.cast(window_function,'float32') * (1 + tf.random_uniform([1],minval=-strech,maxval=strech))