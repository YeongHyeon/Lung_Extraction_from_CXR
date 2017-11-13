import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

def convolution(inputs=None, filters=32, k_size=3, stride=1, padding="same"):

    """https://www.tensorflow.org/api_docs/python/tf/layers/conv1d"""

    # initializers.xavier_initializer()
    # tf.contrib.keras.initializers.he_normal()

    conv = tf.contrib.layers.conv2d(
    inputs=inputs,
    num_outputs=filters,
    kernel_size=k_size,
    stride=stride,
    padding=padding,
    data_format=None,
    rate=1,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=tf.contrib.keras.initializers.he_normal(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
    )

    print("Convolution: "+str(conv.shape))
    return conv

def maxpool(inputs=None, pool_size=2):

    """https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d"""

    maxp = tf.contrib.layers.max_pool2d(
    inputs=inputs,
    kernel_size=pool_size,
    stride=pool_size,
    padding='VALID',
    outputs_collections=None,
    scope=None
    )

    print("Max Pool: "+str(maxp.shape))
    return maxp

def avgpool(inputs=None, pool_size=2):

    """https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d"""

    avg = tf.contrib.layers.avg_pool2d(
    inputs=inputs,
    kernel_size=pool_size,
    stride=pool_size,
    padding='VALID',
    outputs_collections=None,
    scope=None
    )

    print("Average Pool: "+str(avg.shape))
    return avg

def flatten(inputs=None):

    """https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten"""

    flat = tf.contrib.layers.flatten(inputs=inputs)

    print("Flatten: "+str(flat.shape))
    return flat

def fully_connected(inputs=None, num_outputs=None, activate_fn=None):

    """https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected"""

    full_con = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=num_outputs, activation_fn=activate_fn)

    print("Fully Connected: "+str(full_con.shape))
    return full_con

def batch_normalization(inputs=None):
    batchnorm = tf.contrib.layers.batch_norm(
    inputs=inputs,
    decay=0.999,
    center=True,
    scale=False,
    epsilon=0.001,
    activation_fn=None,
    param_initializers=None,
    param_regularizers=None,
    updates_collections=tf.GraphKeys.UPDATE_OPS,
    is_training=True,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    batch_weights=None,
    fused=None,
    data_format=DATA_FORMAT_NHWC,
    zero_debias_moving_mean=False,
    scope=None,
    renorm=False,
    renorm_clipping=None,
    renorm_decay=0.99
    )

    return batchnorm

def dropout(inputs=None, ratio=0.5, train=None):

    """https://www.tensorflow.org/api_docs/python/tf/layers/dropout"""

    drop = tf.layers.dropout(
    inputs=inputs,
    rate=ratio,
    noise_shape=None,
    seed=None,
    training=train,
    name=None
    )

    print("Dropout: "+str(ratio))
    return drop

def convolution_neural_network(x, y_, training=None, height=None, width=None, channel=None, classes=None):

    print("\n** Initialize CNN Layers")

    x_data = tf.reshape(x, [-1, height, width, channel])
    print("Input: "+str(x_data.shape))

    conv_1 = convolution(inputs=x_data, filters=4, k_size=5, stride=1, padding="same")
    maxpool_1 = maxpool(inputs=conv_1, pool_size=2)

    conv_2 = convolution(inputs=maxpool_1, filters=8, k_size=5, stride=1, padding="same")
    maxpool_2 = maxpool(inputs=conv_2, pool_size=2)

    conv_3 = convolution(inputs=maxpool_2, filters=16, k_size=5, stride=1, padding="same")
    maxpool_3 = maxpool(inputs=conv_3, pool_size=2)

    conv_4 = convolution(inputs=maxpool_3, filters=32, k_size=5, stride=1, padding="same")
    maxpool_4 = maxpool(inputs=conv_4, pool_size=2)

    conv_5 = convolution(inputs=maxpool_4, filters=32, k_size=5, stride=1, padding="same")
    maxpool_5 = maxpool(inputs=conv_5, pool_size=2)

    drop_1 = dropout(inputs=maxpool_5, ratio=0.5, train=training)

    flatten_layer = flatten(inputs=drop_1)

    full_con = fully_connected(inputs=flatten_layer, num_outputs=classes, activate_fn=None)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=full_con, labels=y_)
    mean_loss = tf.reduce_mean(cross_entropy) # Equivalent to np.mean

    """https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate"""
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(mean_loss)

    prediction = tf.contrib.layers.softmax(full_con) # Want to prediction Use this!
    correct_pred = tf.equal(tf.argmax(full_con, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return train_step, accuracy, mean_loss, prediction
