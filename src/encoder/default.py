'''
This file defines the default encoder.
'''

import tensorflow as tf

'''
This function defines the encoder that extracts features from a single modality.
    model - the architecture being built, needed for hyperparams and summaries
'''
def encoder_block(model, enc_name, input_layer, num_filters, kernel=[3,3],pool=[2,2],pool_stride=2):
    """Define a single encoder block - 2 conv/relu/bn and then max pool"""
    with tf.variable_scope(enc_name):
        # Convolutional Layer 
        # Computes num_filters features using a kernel x kernel filter with ReLU activation.
        # Padding is added to preserve width and height.
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=num_filters,
            kernel_size=kernel,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(model.hyper.l2_scale),
            bias_initializer=tf.zeros_initializer(),
            padding="same",
            activation=None,
            name='conv1'
        )
        
            
        # apply relu BEFORE batch norm 
        conv_relu1 = tf.nn.leaky_relu(conv1, alpha=model.hyper.relu_leakiness)
        
        #apply batch norm on relu output
        conv_relu_bn1 = tf.layers.batch_normalization(
            inputs=conv_relu1,
            #axis=1, <--- axis to be normalised first - we want last as that will be the num features
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=model.is_training,
            name='bn1'
        )
        
            
        # Convolutional Layer 
        # Computes num_filters features using a kernel x kernel filter with ReLU activation.
        # Padding is added to preserve width and height.
        conv2 = tf.layers.conv2d(
            inputs=conv_relu_bn1,
            filters=num_filters,
            kernel_size=kernel,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(model.hyper.l2_scale),
            bias_initializer=tf.zeros_initializer(),
            padding="same",
            activation=None,
            name='conv2'
        )

            
        # apply relu BEFORE batch norm 
        conv_relu2 = tf.nn.leaky_relu(conv2, alpha=model.hyper.relu_leakiness)
        
        #apply batch norm on relu output
        conv_relu_bn2 = tf.layers.batch_normalization(
            inputs=conv_relu2,
            #axis=1, <--- axis to be normalised first - we want last as that will be the num features
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=model.is_training,
            name='bn2'
        )
            
        
        # apply max pool 
        cbn_pool = tf.layers.max_pooling2d(
            inputs=conv_relu_bn2,
            pool_size=pool,
            strides=pool_stride,
            padding="same"
        )
            
        #encoder
        encoder = cbn_pool
        
        # if model specified, do summaries
        if model is not None: 
            model._activation_summary(conv1)
            model._activation_summary(conv_relu1)
            model._activation_summary(conv_relu_bn1)
            model._activation_summary(conv2)        
            model._activation_summary(conv_relu2)
            model._activation_summary(conv_relu_bn2)
            model._activation_summary(cbn_pool)

        return encoder
    
    
'''
This function defines the default encoder - a stacked architecture.
    model - the architecture being built, needed for hyperparams and summaries
'''
def build_encoder(model, image_data, modality_str, num_blocks=4):
    block_outputs = []
    input_tensor = image_data
    for blk in range(1,num_blocks+1):
        block_name = modality_str + '_Encoder_' + str(blk)
        block = encoder_block(
            model=model,
            enc_name=block_name,
            input_layer=input_tensor,
            num_filters=model.design.num_filters,
            kernel=model.design.conv_kernel,
            pool=model.design.pool_sz,
            pool_stride=model.design.pool_str
        )
        input_tensor = block # needed for stacking
        block_outputs.append(block)
    return block_outputs
        