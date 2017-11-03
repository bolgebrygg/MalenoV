# RotationXY
import tensorflow as tf
def randomRotationXY(X, max_rot):
    max_rot = 6.28318530718 / 360 * max_rot #Deg 2 rad
    theta = tf.random_uniform([1], minval=-max_rot, maxval=max_rot, dtype='float32')
    x = X[2] * tf.cos(theta) - X[1] * tf.sin(theta)
    y = X[2] * tf.sin(theta) + X[1] * tf.cos(theta)
    return tf.stack([X[0],y,x])