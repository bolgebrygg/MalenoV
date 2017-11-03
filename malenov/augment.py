### ---- Functions for data augmentation ---- (Needs further development)
# RotationXY
def randomRotationXY(X, max_rot):
    max_rot = 6.28318530718 / 360 * max_rot #Deg 2 rad
    theta = tf.random_uniform([1], minval=-max_rot, maxval=max_rot, dtype='float32')
    x = X[2] * tf.cos(theta) - X[1] * tf.sin(theta)
    y = X[2] * tf.sin(theta) + X[1] * tf.cos(theta)
    return tf.stack([X[0],y,x])


# RotationZ
def randomRotationZ(X, max_rot):
    max_rot = 6.28318530718 / 360 * max_rot  # Deg 2 rad
    theta = tf.random_uniform([1], minval=-max_rot, maxval=max_rot, dtype='float32')
    t = X[0] * tf.cos(theta) - X[1] * tf.sin(theta)
    x = X[0] * tf.sin(theta) + X[1] * tf.cos(theta)
    return tf.stack([t,x,X[2]])


# Stretching
def randomStretch(window_function, strech):
    return tf.cast(window_function,'float32') * (1 + tf.random_uniform([1],minval=-strech,maxval=strech))


# Flip
def randomFlip(window_function):
    should_flip = tf.cast(tf.random_uniform([1], 0, 2, dtype=tf.int32)[0] > 0, tf.bool)
    window_function = tf.reverse(window_function, tf.pack([should_flip]))
    return window_function