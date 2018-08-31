'''
This file defines the scaled cross-entropy loss function.
'''

import tensorflow as tf

'''
This is the loss function that should be called.
    logits - logits calculated by decoder
    label - ground truth label (binary)
    num_classes - number of classes INCLUDING ground truth
    loss_str - a label for the losses
    
    returns - cross entropy loss
'''
def loss(logits, label, num_classes, loss_str):
    # Flatten the logits, so that we can compute cross-entropy for each pixel and get a sum of cross-entropies.
    flat_logits = tf.reshape(tensor=logits, shape=(-1, num_classes),name='flat_logits')
    flat_label = tf.reshape(tensor=label, shape=(-1, num_classes),name='flat_label')
        
    num_true = tf.count_nonzero(label, axis=[1,2], keepdims=True, name='num_true') # pixels of each class [1,2 are width/height]
    total_true = tf.reduce_sum(num_true, axis=[1,2,3], keepdims=True, name='ttl_true') # total true, calculated per batch
    scaling = 1 - num_true/total_true
    scaling = tf.cast(scaling, tf.float32,name='scaling')
        
    weighted_labels = tf.multiply(label, scaling, 'weighted_labels')
    flat_weights = tf.reshape(tensor=weighted_labels, shape=(-1, num_classes),name='flat_weights')
    weight_map = tf.reduce_sum(flat_weights, axis=1,name='weight_map')
    
        
    # avoid softmax that are 0 using log with an epsilon 
    softmax_val = tf.nn.softmax(flat_logits)
    soft_log = tf.log(softmax_val + 1e-9)
    unweighted_losses = -tf.reduce_sum(flat_label * soft_log, axis=1) 
    
    weighted_losses = tf.multiply(unweighted_losses, weight_map)
    cross_entropy_str='xentropy_mean_' + loss_str
    cross_entropy_mean = tf.reduce_mean(weighted_losses, name=cross_entropy_str)
    tf.add_to_collection('losses', cross_entropy_mean)
    tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
        
    return cross_entropy_mean