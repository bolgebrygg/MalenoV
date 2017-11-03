# Flip
import tensorflow as tf
def randomFlip(window_function):
    should_flip = tf.cast(tf.random_uniform([1], 0, 2, dtype=tf.int32)[0] > 0, tf.bool)
    window_function = tf.reverse(window_function, tf.pack([should_flip]))
    return window_function