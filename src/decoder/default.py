'''
This file defines the default decoder.
'''

import tensorflow as tf

'''
This function defines the decoder block that produces an output from fused information.
    model - the architecture being built, needed for hyperparams and summaries
'''
def decoder_block(model, dec_name, input_layer, num_filters, kernel=[3,3], deconv_stride=1):
    """Define a single decoder block"""
    with tf.variable_scope(dec_name):        
        # instead of transposed conv, which can have artifacts, we will use nearest neighbour resize
        width = input_layer.get_shape()[1]
        height = input_layer.get_shape()[2]
        new_size = [2*width, 2*height]
        ups = tf.image.resize_images(
            images=input_layer,
            size=new_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        
            
        # then a 1 stride deconv to smooth
        deconv = tf.layers.conv2d_transpose(
            inputs=ups,
            filters=num_filters,
            strides=deconv_stride,
            kernel_size=kernel,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(model.hyper.l2_scale),
            bias_initializer=tf.zeros_initializer(),
            padding="same",
            activation=None,
            name='deconv2'
        )
        deconv_relu1 = tf.nn.leaky_relu(deconv, alpha=model.hyper.relu_leakiness)
    
            
        # Convolutional Layer 
        # Computes num_filters features using a kernel x kernel filter with ReLU activation.
        # Padding is added to preserve width and height.
        conv1 = tf.layers.conv2d(
            inputs=deconv,
            filters=num_filters,
            kernel_size=kernel,
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
            padding="same",
            activation=None,
            name='conv'
        )
        
            
        
        # apply relu BEFORE batch norm 
        conv_relu2 = tf.nn.leaky_relu(conv2, alpha=model.hyper.relu_leakiness)
        model._activation_summary(conv_relu2)
        
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
        
        
        #decoder
        decoder = conv_relu_bn2
        
        # if model specified, do summaries
        if model is not None: 
            model._activation_summary(ups)
            model._activation_summary(deconv)
            model._activation_summary(deconv_relu1)
            model._activation_summary(conv1)
            model._activation_summary(conv_relu1)
            model._activation_summary(conv_relu_bn1)
            model._activation_summary(conv2)
            model._activation_summary(conv_relu_bn2)
        return decoder

    
    
    
'''
This function defines the reconstruction block that produces the final output/logits 
    model - the architecture being built, needed for hyperparams and summaries
'''
def reconstruction_block(model, recon_name, input_layer, num_filters, kernel=[1,1]):
    """Define a single region block"""
    with tf.variable_scope(recon_name):
        # Convolutional Layer 
        # Computes num_filters features using a kernel x kernel filter with ReLU activation.
        # Padding is added to preserve width and height.
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=num_filters,
            kernel_size=kernel,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(model.hyper.l2_scale),
            bias_initializer=tf.zeros_initializer(),
            padding="same",
            activation=None,
            name='conv'
        )
        
            
        # apply relu BEFORE batch norm 
        conv_relu = tf.nn.leaky_relu(conv, alpha=model.hyper.relu_leakiness)            
        
        
        #apply batch norm on relu output
        conv_relu_bn = tf.layers.batch_normalization(
            inputs=conv_relu,
            #axis=1, <--- axis to be normalised first - we want last as that will be the num features
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=model.is_training,
            name='bn'
        )
        
            
        #reconstruction
        reconstruction = conv_relu_bn
        
        if model is not None:
            model._activation_summary(conv)
            model._activation_summary(conv_relu)
            model._activation_summary(conv_relu_bn)
            
        return reconstruction
    
 
'''
This function defines the default decoder - a stacked upsampling process.
    model - the architecture being built, needed for hyperparams and summaries
'''
def build_decoder(model, encoders, fusion_maps, num_blocks):
    num_modalities = len(encoders)
    prior = None
    
    # loop fuses and stacks in reverse order of encoder stacks
    for blk in range(num_blocks,0,-1):
        block_name = 'Decoder_' + str(blk)
        
        # first get the encoders of the same level from each modality
        encoders_for_this_block = []        
        for enc in range(num_modalities):
            encoders_for_this_block.append(encoders[enc][blk-1])
        
        # then stack and multiply with the fusion map, then stack with any PRIOR output
        fused = tf.multiply(tf.concat(encoders_for_this_block, axis=-1), fusion_maps[blk-1], name='fused')
        if prior is None:
            encoded = fused
        else:
            encoded = tf.concat([fused, prior], axis=-1)
        
        block = decoder_block(
            model = model,
            dec_name = block_name,
            input_layer=encoded,
            num_filters=model.design.num_filters,
            kernel=model.design.conv_kernel,
            deconv_stride=model.design.deconv_str
        )
        prior = block
        
    # now we should be back to original input w and h dimensions, so put in a reconstruction block (applied a 1x1 conv)
    decoded = reconstruction_block(
        model = model,
        recon_name = 'Final_Decoded',
        input_layer=prior, # uses final decoder
        num_filters=model.num_all_classes, 
        kernel=[1,1] # will use default stride
    )
    probability_map = tf.nn.softmax(decoded, name='probability_map') 
    return decoded, probability_map