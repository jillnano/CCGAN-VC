import tensorflow as tf

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def residual1d_block(
    inputs,
    source_id,
    target_id,
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residual_block_'):
    
    
    id_vectors = tf.concat([source_id, target_id], axis = -1)
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1 = id_bias_add_1d(h1, id_vectors)
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_gates = id_bias_add_1d(h1_gates, id_vectors)
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')   
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2 = id_bias_add_1d(h2, id_vectors)
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs,
    source_id,
    filters, 
    kernel_size,
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1 = id_bias_add_1d(h1, source_id)
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_gates = id_bias_add_1d(h1_gates, source_id)
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample2d_block(
    inputs,
    target_id,
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1 = id_bias_add_2d(h1, target_id)
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_gates = id_bias_add_2d(h1_gates, target_id)
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs,
    target_id,
    filters, 
    kernel_size, 
    strides,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1 = id_bias_add_1d(h1, target_id)
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_gates = id_bias_add_1d(h1_gates, target_id)
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs
    
def id_bias_add_1d(inputs, id):
    num_neuron = inputs.shape.dims[2].value
    id_reshaped = tf.reshape(id, [1, 1, id.shape.dims[-1].value])
    bias = tf.layers.dense(inputs = id_reshaped, units = num_neuron)
    bias_tiled = tf.tile(bias, [1, tf.shape(inputs)[1], 1])
    inputs_bias_added = inputs + bias_tiled
    return inputs_bias_added

def id_bias_add_2d(inputs, id):
    num_neuron = inputs.shape.dims[3].value
    id_reshaped = tf.reshape(id, [1, 1, 1, id.shape.dims[-1].value])
    bias = tf.layers.dense(inputs = id_reshaped, units = num_neuron)
    bias_reshaped = tf.reshape(bias, [1, 1, 1, tf.shape(inputs)[3]])
    bias_tiled = tf.tile(bias_reshaped , [1, tf.shape(inputs)[1], tf.shape(inputs)[2], 1])
    inputs_bias_added = inputs + bias_tiled
    return inputs_bias_added
    
def generator_gatedcnn(inputs, source_id, target_id, reuse = False, scope_name = 'generator_gatedcnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv') # 1 128 128
        h1 = id_bias_add_1d(h1, source_id)
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_gates = id_bias_add_1d(h1_gates, source_id)
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, source_id = source_id, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_') # 1 64 256
        d2 = downsample1d_block(inputs = d1, source_id = source_id, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_') # 1 32 512

        # Residual blocks
        r1 = residual1d_block(inputs = d2, source_id = source_id, target_id = target_id, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_') # 1 32 512
        r2 = residual1d_block(inputs = r1, source_id = source_id, target_id = target_id, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_') # 1 32 512
        r3 = residual1d_block(inputs = r2, source_id = source_id, target_id = target_id, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_') # 1 32 512
        r4 = residual1d_block(inputs = r3, source_id = source_id, target_id = target_id, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_') # 1 32 512
        r5 = residual1d_block(inputs = r4, source_id = source_id, target_id = target_id, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_') # 1 32 512
        r6 = residual1d_block(inputs = r5, source_id = source_id, target_id = target_id, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_') # 1 32 512

        # Upsample
        u1 = upsample1d_block(inputs = r6, target_id = target_id, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_') # 1 64 512
        u2 = upsample1d_block(inputs = u1, target_id = target_id, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_') # 1 128 256

        # Output
        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv') # 1 128 24
        o1 = id_bias_add_1d(o1, target_id)
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2
    
def discriminator(inputs, target_id, num_speakers, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
        h1 = id_bias_add_2d(h1, target_id)
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates')
        h1_gates = id_bias_add_2d(h1_gates, target_id)
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, target_id = target_id, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        d2 = downsample2d_block(inputs = d1, target_id = target_id, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        d3 = downsample2d_block(inputs = d2, target_id = target_id, filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample2d_block3_')

        # Output
        o1 = tf.layers.dense(inputs = d3, units = num_speakers)
        o1 = id_bias_add_2d(o1, target_id)
        o1 = tf.nn.sigmoid(o1)
        
        return o1
