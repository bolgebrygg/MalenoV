# RotationZ
import tensorflow as tf
def randomRotationZ(X, max_rot):
    max_rot = 6.28318530718 / 360 * max_rot  # Deg 2 rad
    theta = tf.random_uniform([1], minval=-max_rot, maxval=max_rot, dtype='float32')
    t = X[0] * tf.cos(theta) - X[1] * tf.sin(theta)
    x = X[0] * tf.sin(theta) + X[1] * tf.cos(theta)
    return tf.stack([t,x,X[2]])