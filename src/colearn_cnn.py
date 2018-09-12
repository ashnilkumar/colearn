"""Co-Learning Convolutional Neural Network built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import tensorflow as tf

import encoder.default as ed
import decoder.default as dd
import colearner.default as cd
import losses.scaled_cross_entropy_loss as ls

import sys

tf.logging.set_verbosity(tf.logging.INFO)

HParams = namedtuple('HParams',
                     'batch_size, num_classes, num_epochs, lrn_rate, lrn_rate_decay_steps, lrn_rate_decay_rate,'
                     'weight_decay_rate, relu_leakiness, optimizer, l2_scale')
DParams = namedtuple('DParams',
                     'width, height, depth, num_mods, num_filters, conv_kernel, pool_sz, pool_str, colearn_kernel, deconv_str,'
                     'num_blocks')


class CoLearnNet(object):
    
    def __init__(self, hyper, design, modalities, modality_names, label):
        self.hyper = hyper
        self.design = design
        
        self.data = []
        for i in range(self.design.num_mods):
            data = tf.identity(modalities[i], name=modality_names[i]) # identity workaround to get tensor name for loader
            self.data.append(data)
        self.modality_names = modality_names
        self.label = tf.identity(label, name='label')
        self.is_training = tf.placeholder(tf.bool, name='train_mode')
        self.num_all_classes = self.label.get_shape().as_list()[3]
        
        

    def _decay(self):
        """L2 weight decay loss."""
        costs = []

        for var in tf.trainable_variables():
            if var.op.name.find(r'conv') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hyper.weight_decay_rate, tf.add_n(costs))
    
    
    def count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fK" % (total_parameters / 1e3))
        
    
    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = x.op.name
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        mean = tf.reduce_mean(x)
        tf.summary.scalar(tensor_name + '/mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
        tf.summary.scalar(tensor_name + '/stddev', stddev)
        tf.summary.scalar(tensor_name + '/max', tf.reduce_max(x))
        tf.summary.scalar(tensor_name + '/min', tf.reduce_min(x))
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

        
        
    def _build_valid_op(self):
        """Build a validation op that is on the graph."""
        self.val_loss = self.loss            
        tf.summary.scalar('validation_cost', self.val_loss)
        self.val_op = self.val_loss

        
        
    def _build_train_op(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for BatchNorm
        with tf.control_dependencies(update_ops): # for BatchNorm
            """Build training specific ops for the graph."""
            self.lrn_rate = tf.train.exponential_decay(self.hyper.lrn_rate, 
                                                       self.global_step,
                                                       self.hyper.lrn_rate_decay_steps,
                                                       self.hyper.lrn_rate_decay_rate, 
                                                       staircase=True, name='learning_rate')
            tf.summary.scalar('learning_rate', self.lrn_rate)

            trainable_variables = tf.trainable_variables()

            if self.hyper.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hyper.optimizer == 'mom':
                optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            self.train_op = optimizer.minimize(self.loss, global_step = self.global_step)

        
    
    def build_colearn_architecture(self):
        self.global_step = tf.train.get_or_create_global_step()
        
        # encode each modality
        encoders = []
        for m in range(self.design.num_mods):
            encoder = ed.build_encoder(self, self.data[m], self.modality_names[m], num_blocks=self.design.num_blocks)
            encoders.append(encoder)
        
        # obtain fusion maps
        fusion_maps = []
        for i in range(self.design.num_blocks):
            colearn_name = 'COLEARN_' + str(i + 1)
            
            # get encoders for this block for each modality
            input_encoders =  []
            for m in range(self.design.num_mods):
                input_encoders.append(encoders[m][i])
            
            # now fiuse
            fusion_map = cd.colearn_unit(self,
                                        colearn_name=colearn_name,
                                        input_modalities = input_encoders,
                                        num_filters=self.design.num_filters,
                                        kernel=self.design.colearn_kernel)
            fusion_maps.append(fusion_map)
            
        # decode
        logits, probability_map = dd.build_decoder(self, 
                                                encoders = encoders,
                                                fusion_maps = fusion_maps,
                                                num_blocks = self.design.num_blocks)

        # put probabilities on the main model for easier access by external classes
        self.probability_map = probability_map
        
        # get losses!
        self.loss = self.obtain_total_loss(logits)
        
        # operations for training and test
        self._build_train_op()
        self._build_valid_op()
        
        # merge summaries for tensorboard
        self.summaries = tf.summary.merge_all()
        
        return
    
    
    def obtain_total_loss(self, logits):
        # loss from logits
        scaled_loss = ls.loss(logits,
                            label = self.label,
                            num_classes = self.num_all_classes,
                            loss_str = 'all')
        
        # regularization loss       
        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization', reg_loss)
        
        # weight decay loss
        decay = self._decay()

        # combined loss here!
        loss = scaled_loss + reg_loss + decay          
        tf.summary.scalar('loss', loss)
        return loss
        
        
def main(unused_argv):
    print('RUNNING DEFAULT MAIN - ONLY FOR CHECKING ARCHITECTURE GRAPH')
    default_design = DParams(width = 256, height = 256, depth = 1, num_mods = 2, num_filters = 64, conv_kernel = [3,3],
                            pool_sz = [2,2], pool_str = 2, colearn_kernel = [2,3,3], deconv_str = 1, num_blocks = 4)
    
    default_hyper = HParams(batch_size=128,
                             num_classes=4,
                             num_epochs=1,
                             lrn_rate=0.1,
                             lrn_rate_decay_steps=1000,
                             lrn_rate_decay_rate=0.1,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom',
                             l2_scale=0.1)
    
    
    default_ct = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_ct')
    default_pt = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_pet')
    
    default_ctlb = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,3], name='default_ctlb')
    default_ptlb = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_ptlb')
    default_bglb = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_bglb')
    default_label = tf.concat([default_ctlb, default_ptlb, default_bglb], axis = 3, name = 'default_label')
    default_modality_names = ['CT', 'PET']
    
    net = CoLearnNet(default_hyper, default_design, [default_ct,default_pt], default_modality_names, default_label)
    net.build_colearn_architecture()
    net.count_trainable_params()
    print('ARCHITECTURE GRAPH SEEMS OK')
    

if __name__ == "__main__":
    tf.app.run()