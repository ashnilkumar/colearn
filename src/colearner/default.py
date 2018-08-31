'''
This file defines the default co-learning unit.
'''

import tensorflow as tf

'''
This function transforms the multi-modality inputs together for 3D convolution
    inputs - list of input modalities
    
    returns - stacked, padded and transposed tensor for 3D convolution
'''
def transform_mm_input(inputs):
    """Stacks the input tensors that go into the 3D convolution"""
    # first stack all inputs
    input_mm=tf.stack(inputs,4)
        
    # we need to pad 0s around w and h to allow for 3x3 colearning 3d conv
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]]) # only pads dim 2 and 3 (h and w)
    input_mm = tf.pad(input_mm, paddings, "CONSTANT")
        
    # inputs are in "[batch, height, width, num_filt] for each modality" need them to be "[batch, modality, height, width, num_filt]" which is the input form for conv3d
        
    input_mm=tf.transpose(input_mm,[0,4,1,2,3])
    return input_mm


'''
This function defines the co-learning unit that performs 3D convolution across different modalities.
    model - the architecture being built, needed for hyperparams and summaries
    colearn_name - name of the op
    input_modalities - list of tensors of each modality
    num_filters - number of filters to obtain PER modality
    kernel - the 3D kernel to use - first dimension MUST be equal to the number of modalities
    
    returns - the fusion map
'''
def colearn_unit(model, colearn_name, input_modalities, num_filters, kernel=[2,3,3]):
    mm_stack = transform_mm_input(input_modalities)
    num_modalities = len(input_modalities)
    
    """Define a colearning convolution"""
    with tf.variable_scope(colearn_name):
        # Convolutional Layer 
        # Computes num_filters features using a kernel x kernel filter with ReLU activation.
        # Padding is added to preserve width and height.
        colearn = tf.layers.conv3d(
            inputs=mm_stack,
            filters=(num_filters*num_modalities), # multiplied by num_modalities so don't need a broadcast later
            kernel_size=kernel, 
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(model.hyper.l2_scale),
            bias_initializer=tf.zeros_initializer(),
            padding="valid", # input has been padded around w and h so this reduces it to singleton modality
            activation=tf.nn.leaky_relu,
            name='conv3d'
        )
        if model is not None:             
            model._activation_summary(colearn)
        colearn_out = tf.squeeze(colearn,1)
        return colearn_out
    
