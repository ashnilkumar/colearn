"""Cross-Modality Convolutional Neural Network built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import tensorflow as tf

import sys

tf.logging.set_verbosity(tf.logging.INFO)

HParams = namedtuple('HParams',
                     'batch_size, num_classes, num_epochs, min_lrn_rate, lrn_rate, lrn_rate_decay_steps, lrn_rate_decay_rate,'
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer, l2_scale, loss_params')
DParams = namedtuple('DParams',
                     'width, height, depth, num_mods, num_filters, conv_kern, pool_sz, pool_str, cross_mod_kern, deconv_str,'
                     'cost,output_style')

# define loss constants
CROSS_ENTROPY = 'cross_entropy'
WEIGHTED_CROSS_ENTROPY = 'weighted_cross_entropy'
WEIGHTED_SCALED_CROSS_ENTROPY = 'weighted_scaled_cross_entropy'
SCALED_CROSS_ENTROPY = 'scaled_cross_entropy'
DICE_COEFFICIENT = 'dice_coefficient'
DICE_DROP_BG = 'dice_drop_bg'
SCALED_DICE = 'scaled_dice'
BINARY_DICE = 'binary_dice'
SCALED_BINARY_DICE = 'scaled_binary_dice'
ACCURACY = 'accuracy'
SCALED_ACCURACY = 'scaled_accuracy'
PRECISION = 'precision'
SCALED_PRECISION = 'scaled_precision'

# define output styles
STYLE_SPLIT = 'split'
STYLE_SINGLE = 'single'
STYLE_DEFAULT = STYLE_SINGLE


class PreFusedNet(object):
    
    def __init__(self, hyper, design, ct, pt, lbct, lbpt, lbbg, mode):
        self.hyper = hyper
        self.design = design
        #self.ct = ct
        #self.pt = pt
        #self.lbct = lbct
        #self.lbpt = lbpt
        #self.lbbg = lbbg
        self.ct = tf.identity(ct, name='ct') # identity used as workaround to get tensor name to work with loader
        self.pt = tf.identity(pt, name='pt')
        batch_fused = tf.add(self.ct, self.pt)
        self.fused = tf.map_fn(lambda petct_pair: tf.image.per_image_standardization(petct_pair), batch_fused)
        self.fused = tf.identity(self.fused, name='fused')
        self.lbct = tf.identity(lbct, name='lbct')
        self.lbpt = tf.identity(lbpt, name='lbpt')
        self.lbbg = tf.identity(lbbg, name='lbbg')
        self.lb_pos_gt = tf.concat(values=[lbct, lbpt], axis=3, name='lb_pos_gt') # create a combined ground truth for both pet and ct
        self.mode = mode
        #if mode ==' train':
        #    self.is_training = True
        #else:
        #    self.is_training = False
        self.is_training = tf.placeholder(tf.bool, name='train_mode')

        self.num_ct_classes = lbct.get_shape().as_list()[3]
        self.num_pt_classes = lbpt.get_shape().as_list()[3]
        self.num_all_classes = self.num_ct_classes + self.num_pt_classes
        
        self._extra_train_ops = []
        

    def transform_mm_input(self, inputs):
        """Stacks the input tensors that go into the cross-modality convolution"""
        # first stack all inputs
        input_mm=tf.stack(inputs,4)
        
        # we need to pad 0s around w and h to allow for 3x3 colearning 3d conv
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]]) # only bads dim 2 and 3 (h and w)
        input_mm = tf.pad(input_mm, paddings, "CONSTANT")
        
        # inputs are in "[batch, height, width, num_filt] for each modality" need them to be "[batch, modality, height, width, num_filt]" which is the input form for conv3d
        
        input_mm=tf.transpose(input_mm,[0,4,1,2,3])
        return input_mm

    
    def cross_mod_conv(self, cmc_name,input_mm,num_filters,kernel=[2,1,1]):
        """Define a cross modality convolution"""
        #print('CREATING CMC')
        with tf.variable_scope(cmc_name):
            # Convolutional Layer 
            # Computes num_filters features using a kernel x kernel filter with ReLU activation.
            # Padding is added to preserve width and height.
            cmc = tf.layers.conv3d(
                inputs=input_mm,
                filters=(num_filters*2), ### multiplied by 2 so don't need a broadcast later
                kernel_size=kernel, 
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hyper.l2_scale),
                bias_initializer=tf.zeros_initializer(),
                padding="valid", # input has been padded around w and h so this reduces it to singleton modality
                activation=tf.nn.leaky_relu,
                name='conv3d'
            )
            self._activation_summary(cmc)
            cmc_out = tf.squeeze(cmc,1)
            return cmc_out
            #return cmc
    
    
    def region_block(self, reg_name, input_layer, num_filters, kernel=[1,1]):
        """Define a single region block"""
        #print('CREATING REGION BLOCK')
        with tf.variable_scope(reg_name):
            # Convolutional Layer 
            # Computes num_filters features using a kernel x kernel filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv = tf.layers.conv2d(
                inputs=input_layer,
                filters=num_filters,
                kernel_size=kernel,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hyper.l2_scale),
                bias_initializer=tf.zeros_initializer(),
                padding="same",
                activation=None,
                name='conv'
            )
            self._activation_summary(conv)
            
            # apply relu BEFORE batch norm - https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            #conv_relu = tf.nn.leaky_relu(conv, alpha=self.hyper.relu_leakiness)
            conv_relu = tf.nn.leaky_relu(conv)
            self._activation_summary(conv_relu)
        
            #apply batch norm on relu output
            conv_relu_bn = tf.layers.batch_normalization(
                inputs=conv_relu,
                #axis=1, <--- axis to be normalised first - we want last as that will be the num features
                momentum=0.9,
                epsilon=0.001,
                center=True,
                scale=True,
                training=self.is_training,
                name='bn'
            )
            self._activation_summary(conv_relu_bn)
            
            #region
            region = conv_relu_bn
            
        return region
    
    
    def decoder_block(self, dec_name, input_layer, num_filters, kernel=[3,3], deconv_stride=2):
        """Define a single decoder block"""
        #print('CREATING DECODER')
        with tf.variable_scope(dec_name):
            # first we deconv using conv_transpose
            #deconv = tf.layers.conv2d_transpose(
            #    inputs=input_layer,
            #    filters=num_filters,
            #    strides=deconv_stride,
            #    kernel_size=kernel,
            #    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hyper.l2_scale),
            #    bias_initializer=tf.zeros_initializer(),
            #    padding="same",
            #    activation=None,
            #    name='deconv'
            #)
            #self._activation_summary(deconv)
            
            
            # instead of transposed conv, which can have artifacts, we will use nearest neighbour resize
            width = input_layer.get_shape().as_list()[1]
            height = input_layer.get_shape().as_list()[2]
            new_size = [2*width, 2*height]
            deconv = tf.image.resize_images(
                images=input_layer,
                size=new_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            self._activation_summary(deconv)
            
            # then a 1 stride deconv to smooth
            deconv2 = tf.layers.conv2d_transpose(
                inputs=deconv,
                filters=num_filters,
                strides=1,
                kernel_size=kernel,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hyper.l2_scale),
                bias_initializer=tf.zeros_initializer(),
                padding="same",
                activation=None,
                name='deconv2'
            )
            self._activation_summary(deconv2)
            deconv_relu1 = tf.nn.leaky_relu(deconv2, alpha=self.hyper.relu_leakiness)
            self._activation_summary(deconv_relu1)
            
            
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
            self._activation_summary(conv1)
            
       
        
            # apply relu BEFORE batch norm - https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            conv_relu1 = tf.nn.leaky_relu(conv1, alpha=self.hyper.relu_leakiness)
            self._activation_summary(conv_relu1)
        
            #apply batch norm on relu output
            conv_relu_bn1 = tf.layers.batch_normalization(
                inputs=conv_relu1,
                #axis=1, <--- axis to be normalised first - we want last as that will be the num features
                momentum=0.9,
                epsilon=0.001,
                center=True,
                scale=True,
                training=self.is_training,
                name='bn1'
            )
            self._activation_summary(conv_relu_bn1)
            
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
            self._activation_summary(conv2)
            
        
            # apply relu BEFORE batch norm - https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            conv_relu2 = tf.nn.leaky_relu(conv2, alpha=self.hyper.relu_leakiness)
            self._activation_summary(conv_relu2)
        
            #apply batch norm on relu output
            conv_relu_bn2 = tf.layers.batch_normalization(
                inputs=conv_relu2,
                #axis=1, <--- axis to be normalised first - we want last as that will be the num features
                momentum=0.9,
                epsilon=0.001,
                center=True,
                scale=True,
                training=self.is_training,
                name='bn2'
            )
            self._activation_summary(conv_relu_bn2)
        
            #decoder
            decoder = conv_relu_bn2
            
        return decoder


    def encoder_block(self, enc_name, input_layer, num_filters, kernel=[3,3],pool=[2,2],pool_stride=2):
        """Define a single encoder block - 2 conv/relu/bn and then max pool"""
        #print('CREATING ENCODER')
        with tf.variable_scope(enc_name):
            # Convolutional Layer 
            # Computes num_filters features using a kernel x kernel filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=num_filters,
                kernel_size=kernel,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hyper.l2_scale),
                bias_initializer=tf.zeros_initializer(),
                padding="same",
                activation=None,
                name='conv1'
            )
            self._activation_summary(conv1)
            
            # apply relu BEFORE batch norm - https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            conv_relu1 = tf.nn.leaky_relu(conv1, alpha=self.hyper.relu_leakiness)
            self._activation_summary(conv_relu1)
        
            #apply batch norm on relu output
            conv_relu_bn1 = tf.layers.batch_normalization(
                inputs=conv_relu1,
                #axis=1, <--- axis to be normalised first - we want last as that will be the num features
                momentum=0.9,
                epsilon=0.001,
                center=True,
                scale=True,
                training=self.is_training,
                name='bn1'
            )
            self._activation_summary(conv_relu_bn1)
        
            
            # Convolutional Layer 
            # Computes num_filters features using a kernel x kernel filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv2 = tf.layers.conv2d(
                inputs=conv_relu_bn1,
                filters=num_filters,
                kernel_size=kernel,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hyper.l2_scale),
                bias_initializer=tf.zeros_initializer(),
                padding="same",
                activation=None,
                name='conv2'
            )
            self._activation_summary(conv2)
            
            # apply relu BEFORE batch norm - https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            conv_relu2 = tf.nn.leaky_relu(conv2, alpha=self.hyper.relu_leakiness)
            self._activation_summary(conv_relu2)
        
            #apply batch norm on relu output
            conv_relu_bn2 = tf.layers.batch_normalization(
                inputs=conv_relu2,
                #axis=1, <--- axis to be normalised first - we want last as that will be the num features
                momentum=0.9,
                epsilon=0.001,
                center=True,
                scale=True,
                training=self.is_training,
                name='bn2'
            )
            self._activation_summary(conv_relu_bn2)
            
        
            # apply max pool 
            cbn_pool = tf.layers.max_pooling2d(
                inputs=conv_relu_bn2,
                pool_size=pool,
                strides=pool_stride,
                padding="same"
            )
            self._activation_summary(cbn_pool)
            
            #encoder
            encoder = cbn_pool
            
        return encoder
    
    
    def construct_network(self, input_ct,input_pet,input_fuse):
        """Define the full network"""
        print('CONSTRUCTING NETWORK')
        
        # ENCODER LEVEL 1
        encoder1 = self.encoder_block(
            "Encoder_1",
            input_layer=input_fuse,
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            pool=self.design.pool_sz,
            pool_stride=self.design.pool_str
        )
        
        
        
        # ENCODER LEVEL 2
        encoder2 = self.encoder_block(
            "Encoder_2",
            input_layer=encoder1,
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            pool=self.design.pool_sz,
            pool_stride=self.design.pool_str
        )
        
        
        
        encoder3 = self.encoder_block(
            "Encoder_3",
            input_layer=encoder2,
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            pool=self.design.pool_sz,
            pool_stride=self.design.pool_str
        )
        
        
        
        encoder4 = self.encoder_block(
            "Encoder_4",
            input_layer=encoder3,
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            pool=self.design.pool_sz,
            pool_stride=self.design.pool_str
        )
        
        
        decoder4 = self.decoder_block(
            "decoder_4",
            input_layer=encoder4, 
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            deconv_stride=self.design.deconv_str
        )
        
        #decoder4 = self.decoder_block(
        #    "decoder_4",
        #    input_layer=cmc4, # CMC 
        #    num_filters=self.design.num_filters,
        #    kernel=self.design.conv_kern,
        #    deconv_stride=self.design.deconv_str
        #)
        
        
        decoder3 = self.decoder_block(
            "decoder_3",
            input_layer=decoder4, 
            #input_layer=tf.multiply(decoder4,cmc3), # also should mult by CMC
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            deconv_stride=self.design.deconv_str
        )
        

        decoder2 = self.decoder_block(
            "decoder_2",
            input_layer=decoder3,
            #input_layer=tf.multiply(decoder3,cmc2),# also should mult by CMC
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            deconv_stride=self.design.deconv_str
        )
        

        decoder1 = self.decoder_block(
            "decoder_1",
            input_layer=decoder2, 
            #input_layer=tf.multiply(decoder2,cmc1),# also should mult by CMC
            num_filters=self.design.num_filters,
            kernel=self.design.conv_kern,
            deconv_stride=self.design.deconv_str
        )
        
        ### the next part depends on the output style
        if self.design.output_style == STYLE_SPLIT:
            print('USING SPLIT OUTPUT')
            self.construct_split_output(decoder1)
        elif self.design.output_style == STYLE_SINGLE:
            print('USING SINGLE OUTPUT')
            self.construct_single_output(decoder1)
        elif self.design.output_style == STYLE_DEFAULT:
            print('USING DEFAULT SINGLE OUTPUT')
            self.construct_single_output(decoder1)
        
              
        # constructed!
        return
    
    
    def construct_single_output(self, decoder1):
        # this will give us a tensor of shape=(?, 256, 256, 64) - so we need to refine into PET and CT labels
        # according to U-Net paper, final is a 1x1 Conv: https://arxiv.org/pdf/1505.04597.pdf
        full_decoder = self.region_block(
            "region_all",
            input_layer=decoder1, # uses cross-mod final decoder
            num_filters=self.num_all_classes + 1, # <<<<<<<<<<<<< NEED TO INCLUDE BG 
            kernel=[1,1] # will use default stride
        )
        self.recon_all = full_decoder
        
         # and now define costs, predictions, and probabilities   
        with tf.variable_scope('costs'):
            # basic cross entropy loss
            if self.design.cost == CROSS_ENTROPY:
                # usual cross entropy loss
                print('USING CROSS ENTROPY LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_cross_entropy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')            
            elif self.design.cost == WEIGHTED_CROSS_ENTROPY:
                # weighted cross entropy loss
                print('USING WEIGHTED CROSS ENTROPY LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_weighted_cross_entropy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == WEIGHTED_SCALED_CROSS_ENTROPY:
                print('USING WEIGHTED AND SCALED CROSS ENTROPY LOSS')
                # weighted and scaled cross entropy loss
                self.all_cost, self.all_probabilities, self.all_pred = self.get_weighted_scaled_cross_entropy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == SCALED_CROSS_ENTROPY:
                print('USING SCALED CROSS ENTROPY LOSS')
                # scaled cross entropy loss
                self.all_cost, self.all_probabilities, self.all_pred = self.get_scaled_cross_entropy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')    
            elif self.design.cost == DICE_COEFFICIENT:                
                print('USING DICE COEFFICIENT LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_dice_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == DICE_DROP_BG:
                print('USING DICE COEFFICIENT LOSS DROPPING BACKGROUND')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_dice_loss_dropbg(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == SCALED_DICE:
                print('USING SCALED DICE COEFFICIENT LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_scaled_dice_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == BINARY_DICE:             
                print('USING BINARY DICE COEFFICIENT LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_binary_dice_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == SCALED_BINARY_DICE:
                print('USING SCALED BINARY DICE COEFFICIENT LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_scaled_binary_dice_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == ACCURACY:                
                print('USING ACCURACY LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_accuracy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == SCALED_ACCURACY:
                print('USING SCALED ACCURACY LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_scaled_accuracy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == PRECISION:
                print('USING PRECISION LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_precision_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            elif self.design.cost == SCALED_PRECISION:
                print('USING SCALED PRECISION LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_scaled_precision_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
            else:
                # if no cost use cross entropy by default
                print('USING DEFAULT CROSS ENTROPY LOSS')
                self.all_cost, self.all_probabilities, self.all_pred = self.get_cross_entropy_loss(self.recon_all,
                    self.lb_pos_gt, self.lbbg, self.num_all_classes, 'all')
                
            # regularization loss
            self.reg_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar('regularization', self.reg_loss)

            # combined loss here!
            self.cost = self.all_cost + self.reg_loss
            self.decay = self._decay()
            self.cost += self.decay                       
            tf.summary.scalar('cost', self.cost)
        return
        
        
        
    
    
    def construct_split_output(self, decoder1):
        # this will give us a tensor of shape=(?, 256, 256, 64) - so we need to refine into PET and CT labels
        # according to U-Net paper, final is a 1x1 Conv: https://arxiv.org/pdf/1505.04597.pdf
        ct_decoder = self.region_block(
            "region_ct",
            input_layer=decoder1, # uses cross-mod final decoder
            num_filters=self.num_ct_classes + 1, # <<<<<<<<<<<<< NEED TO INCLUDE BG 
            kernel=[1,1] # will use default stride
        )
        self.recon_ct = ct_decoder
        
        pt_decoder = self.region_block(
            "region_pt",
            input_layer=decoder1, # uses final cross-mod decoder
            num_filters=self.num_pt_classes + 1, # <<<<<<<<<<<<< NEED TO INCLUDE BG 
            kernel=[1,1] # will use default stride
        )
        self.recon_pt = pt_decoder
        
        # and now define costs, predictions, and probabilities   
        with tf.variable_scope('costs'):
            # basic cross entropy loss
            if self.design.cost == CROSS_ENTROPY:
                # usual cross entropy loss
                print('USING CROSS ENTROPY LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_cross_entropy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_cross_entropy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == WEIGHTED_CROSS_ENTROPY:
                # weighted cross entropy loss
                print('USING WEIGHTED CROSS ENTROPY LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_weighted_cross_entropy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_weighted_cross_entropy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == WEIGHTED_SCALED_CROSS_ENTROPY:
                print('USING WEIGHTED AND SCALED CROSS ENTROPY LOSS')
                # weighted and scaled cross entropy loss
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_weighted_scaled_cross_entropy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_weighted_scaled_cross_entropy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == SCALED_CROSS_ENTROPY:
                print('USING SCALED CROSS ENTROPY LOSS')
                # scaled cross entropy loss
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_scaled_cross_entropy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_scaled_cross_entropy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == DICE_COEFFICIENT:
                print('USING DICE COEFFICIENT LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_dice_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_dice_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == DICE_DROP_BG:
                print('USING DICE COEFFICIENT LOSS DROPPING BACKGROUND')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_dice_loss_dropbg(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_dice_loss_dropbg(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == SCALED_DICE:
                print('USING SCALED DICE COEFFICIENT LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_scaled_dice_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_scaled_dice_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == BINARY_DICE:
                print('USING BINARY DICE COEFFICIENT LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_binary_dice_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_binary_dice_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == SCALED_BINARY_DICE:
                print('USING SCALED BINARY DICE COEFFICIENT LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_scaled_binary_dice_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_scaled_binary_dice_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == ACCURACY:
                print('USING ACCURACY LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_accuracy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_accuracy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == SCALED_ACCURACY:
                print('USING SCALED ACCURACY LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_scaled_accuracy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_scaled_accuracy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == PRECISION:
                print('USING PRECISION LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_precision_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_precision_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            elif self.design.cost == SCALED_PRECISION:
                print('USING SCALED PRECISION LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_scaled_precision_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_scaled_precision_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            else:
                # if no cost use cross entropy by default
                print('USING DEFAULT CROSS ENTROPY LOSS')
                self.ct_cost, self.ct_probabilities, self.ct_pred = self.get_cross_entropy_loss(self.recon_ct,
                    self.lbct, self.lbbg, self.num_ct_classes, 'ct')            
                self.pt_cost, self.pt_probabilities, self.pt_pred = self.get_cross_entropy_loss(self.recon_pt,
                    self.lbpt, self.lbbg, self.num_pt_classes, 'pt')
            
            # regularization loss
            self.reg_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar('regularization', self.reg_loss)

            # combined loss here!
            self.cost = self.ct_cost + self.pt_cost + self.reg_loss
            self.decay = self._decay()
            self.cost += self.decay                       
            tf.summary.scalar('cost', self.cost)
        return
        
    
    def get_cross_entropy_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        flat_logits = tf.reshape(tensor=logits, shape=(-1, num_classes + 1))
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        flat_label = tf.reshape(tensor=combined_label, shape=(-1, num_classes + 1))
        
        # avoid softmax that are 0 using log with an epsilon 
        softmax_val = tf.nn.softmax(flat_logits)
        soft_log = tf.log(softmax_val + 1e-9)
        cross_entropy_sum = -tf.reduce_sum(flat_label * soft_log, axis=1) 
        cross_entropy_str='xentropy_mean_' + modality_str
        cross_entropy_mean = tf.reduce_mean(cross_entropy_sum,name=cross_entropy_str)
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
            
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        return cross_entropy_mean, probabilities, prediction
    
    
    def get_weighted_cross_entropy_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        repeated = np.repeat([self.hyper.loss_params], num_classes)
        cross_ent_weights = np.concatenate((repeated, np.array([1 - self.hyper.loss_params])))
        weights = tf.constant(cross_ent_weights, dtype=tf.float32)
        flat_logits = tf.reshape(tensor=logits, shape=(-1, num_classes + 1))
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        weighted_label = tf.multiply(weights, combined_label)
        #flat_label = tf.reshape(tensor=combined_label, shape=(-1, num_classes + 1))
        w_flat_label = tf.reshape(tensor=weighted_label, shape=(-1, num_classes + 1))
        
        # avoid softmax that are 0 using log with an epsilon 
        softmax_val = tf.nn.softmax(flat_logits)
        soft_log = tf.log(softmax_val + 1e-9)
        cross_entropy_sum = -tf.reduce_sum(w_flat_label * soft_log, axis=1) # weighted label here
        #weighted_sum = weights * cross_entropy_sum
        weighted_sum = cross_entropy_sum # use base cross entr sum since weights in label
        cross_entropy_str='xentropy_mean_' + modality_str
        cross_entropy_mean = tf.reduce_mean(weighted_sum,name=cross_entropy_str)
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
            
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        return cross_entropy_mean, probabilities, prediction

    
    def get_weighted_scaled_cross_entropy_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        repeated = np.repeat([self.hyper.loss_params], num_classes)
        cross_ent_weights = np.concatenate((repeated, np.array([1 - self.hyper.loss_params])))
        weights = tf.constant(cross_ent_weights, dtype=tf.float32)
        flat_logits = tf.reshape(tensor=logits, shape=(-1, num_classes + 1))
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        weighted_label = tf.multiply(weights, combined_label)
        #flat_label = tf.reshape(tensor=combined_label, shape=(-1, num_classes + 1))
        w_flat_label = tf.reshape(tensor=weighted_label, shape=(-1, num_classes + 1))
        
        # avoid softmax that are 0 using log with an epsilon 
        softmax_val = tf.nn.softmax(flat_logits)
        soft_log = tf.log(softmax_val + 1e-9)
        cross_entropy_sum = -tf.reduce_sum(w_flat_label * soft_log, axis=1) # weighted label here
        #weighted_sum = weights * cross_entropy_sum
        weighted_sum = cross_entropy_sum # use base cross entr sum since weights in label
        cross_entropy_str='xentropy_mean_' + modality_str
        cross_entropy_mean = tf.reduce_mean(weighted_sum,name=cross_entropy_str)/tf.constant(num_classes.value,dtype=tf.float32) # scale by num of classes
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
            
        probabilities = tf.nn.softmax(logits, name=modality_str+'_probabilities') # not flattened!
        prediction = tf.argmax(probabilities, axis=3, name=modality_str+'_prediction') # which axis has the maximum value
        return cross_entropy_mean, probabilities, prediction
    
    
    def get_scaled_cross_entropy_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        flat_logits = tf.reshape(tensor=logits, shape=(-1, num_classes + 1))
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        flat_label = tf.reshape(tensor=combined_label, shape=(-1, num_classes + 1))
        
        num_true = tf.count_nonzero(combined_label, axis=[1,2], keep_dims=True) # how many pixels of each class [1,2 are width/height]
        total_true = tf.reduce_sum(num_true, axis=[1,2,3], keep_dims=True) # total true, calculate per batch, so wipe out all other axes
        scaling = 1 - num_true/total_true
        scaling = tf.cast(scaling, tf.float32)
        
        weighted_labels = tf.multiply(combined_label, scaling)
        flat_weights = tf.reshape(tensor=weighted_labels, shape=(-1, num_classes + 1))
        weight_map = tf.reduce_sum(flat_weights, axis=1)
        
        # avoid softmax that are 0 using log with an epsilon 
        softmax_val = tf.nn.softmax(flat_logits)
        soft_log = tf.log(softmax_val + 1e-9)
        unweighted_losses = -tf.reduce_sum(flat_label * soft_log, axis=1) 
        #unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=flat_label, logits=flat_logits)
        weighted_losses = tf.multiply(unweighted_losses, weight_map)
        cross_entropy_str='xentropy_mean_' + modality_str
        cross_entropy_mean = tf.reduce_mean(weighted_losses,name=cross_entropy_str)
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
        
        # avoid softmax that are 0 using log with an epsilon 
        #softmax_val = tf.nn.softmax(flat_logits)
        #soft_log = tf.log(softmax_val + 1e-9)
        #cross_entropy_sum = -tf.reduce_sum(flat_label * soft_log, axis=1) 
        #cross_entropy_str='xentropy_mean_' + modality_str
        #cross_entropy_mean = tf.reduce_mean(cross_entropy_sum,name=cross_entropy_str)
        #tf.add_to_collection('losses', cross_entropy_mean)
        #tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
            
        probabilities = tf.nn.softmax(logits, name=modality_str+'_probabilities') # not flattened!
        prediction = tf.argmax(probabilities, axis=3, name=modality_str+'_prediction') # which axis has the maximum value
        return cross_entropy_mean, probabilities, prediction
    
    
    def get_dice_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        intersection = tf.reduce_sum(probabilities * combined_label, axis=axis) # uses probabilities may need binary
        predicted = tf.reduce_sum(probabilities, axis=axis) # uses probabilities may need binary
        expected = tf.reduce_sum(combined_label, axis=axis)
        
        dice = (2. * intersection + smooth) / (predicted + expected + smooth)
        dice_str = 'xdice_mean_' + modality_str
        dice_loss = tf.reduce_mean(1 - dice, name=dice_str) # avg across batch - needed or just change 1-dice to batch_size - dice?
        tf.add_to_collection('losses', dice_loss)
        tf.summary.scalar(dice_str, dice_loss)
        return dice_loss, probabilities, prediction
    
    
    def get_dice_loss_dropbg(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        bg_removed_probabilities = probabilities[:,:,:,-1]
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = modality_gt
        
        intersection = tf.reduce_sum(bg_removed_probabilities * combined_label, axis=axis) # uses probabilities may need binary
        predicted = tf.reduce_sum(bg_removed_probabilities, axis=axis) # uses probabilities may need binary
        expected = tf.reduce_sum(combined_label, axis=axis)
        
        dice = (2. * intersection + smooth) / (predicted + expected + smooth)
        dice_str = 'xdice_mean_' + modality_str
        dice_loss = tf.reduce_mean(1 - dice, name=dice_str) # avg across batch - needed or just change 1-dice to batch_size - dice?
        tf.add_to_collection('losses', dice_loss)
        tf.summary.scalar(dice_str, dice_loss)
        return dice_loss, probabilities, prediction
    
    
    def get_scaled_dice_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        
        smooth = 1e-5
        per_class_dice_axis=[1,2]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        num_true = tf.count_nonzero(combined_label, axis=[1,2], keep_dims=True) # how many pixels of each class [1,2 are width/height]
        total_true = tf.reduce_sum(num_true, axis=[1,2,3]) # total true, calculate per batch, so wipe out all other axes
        scaling = 1 - tf.divide(num_true,total_true)
        scaling = tf.cast(scaling, tf.float32)
        
        intersection = tf.reduce_sum(probabilities * combined_label, axis=per_class_dice_axis) 
        predicted = tf.reduce_sum(probabilities, axis=per_class_dice_axis) 
        expected = tf.reduce_sum(combined_label, axis=per_class_dice_axis)
        
        per_class_dice = (2. * intersection + smooth) / (predicted + expected + smooth) # [batch, num_classes] - consequence of prev
        dice = tf.reduce_sum(per_class_dice * scaling/(num_classes + 1), axis=1) # divide by total classes to ensure max is 1
        dice_str = 'xdice_mean_' + modality_str
        dice_loss = tf.reduce_mean(1 - dice, name=dice_str) # avg across batch - needed or just change 1-dice to batch_size - dice?
        tf.add_to_collection('losses', dice_loss)
        tf.summary.scalar(dice_str, dice_loss)
        return dice_loss, probabilities, prediction
    
    
    def get_binary_dice_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        pred_one_hot = tf.one_hot(prediction, num_classes + 1) # go from label to one hot format to make binary easier
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        intersection = tf.reduce_sum(pred_one_hot * combined_label, axis=axis) # uses probabilities may need binary
        predicted = tf.reduce_sum(pred_one_hot, axis=axis) # uses probabilities may need binary
        expected = tf.reduce_sum(combined_label, axis=axis)
        
        dice = (2. * intersection + smooth) / (predicted + expected + smooth)
        dice_str = 'xdice_mean_' + modality_str
        dice_loss = tf.reduce_mean(1 - dice, name=dice_str) # avg across batch - needed or just change 1-dice to batch_size - dice?
        tf.add_to_collection('losses', dice_loss)
        tf.summary.scalar(dice_str, dice_loss)
        return dice_loss, probabilities, prediction
    
    
    def get_scaled_binary_dice_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        pred_one_hot = tf.one_hot(prediction, num_classes + 1) # go from label to one hot format to make binary easier
        
        smooth = 1e-5
        axis=[1,2]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        num_true = tf.count_nonzero(combined_label, axis=[1,2], keep_dims=True) # how many pixels of each class [1,2 are width/height]
        total_true = tf.reduce_sum(num_true, axis=[1,2,3]) # total true, calculate per batch, so wipe out all other axes
        scaling = 1 - num_true/total_true
        scaling = tf.cast(scaling, tf.float32)
        
        intersection = tf.reduce_sum(pred_one_hot * combined_label, axis=axis) # [batch, num_classes] remains
        predicted = tf.reduce_sum(pred_one_hot, axis=axis) # [batch, num_classes] remains
        expected = tf.reduce_sum(combined_label, axis=axis) # [batch, num_classes] remains
        
        per_class_dice = (2. * intersection + smooth) / (predicted + expected + smooth) # [batch, num_classes] - consequence of prev
        dice = tf.reduce_sum(per_class_dice * scaling/(num_classes + 1), axis=1) # divide by total classes to ensure max is 1
        dice_str = 'xdice_mean_' + modality_str
        dice_loss = tf.reduce_mean(1 - dice, name=dice_str) # avg across batch - needed or just change 1-dice to batch_size - dice?
        tf.add_to_collection('losses', dice_loss)
        tf.summary.scalar(dice_str, dice_loss)
        return dice_loss, probabilities, prediction
    
    
    def get_accuracy_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        pred_one_hot = tf.one_hot(prediction, num_classes + 1) # go from label to one hot format to make binary easier
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
                
        acc_str = 'xacc_mean_' + modality_str
        accuracy, _ = tf.metrics.accuracy(combined_label, pred_one_hot)
        accuracy_loss = tf.reduce_mean(1 - accuracy, name=accc_str)
        tf.add_to_collection('losses', accuracy_loss)
        tf.summary.scalar(acc_str, accuracy_loss)
        return accuracy_loss, probabilities, prediction
    
    
    ###  may need to do accuracy individually for each class and then scale, currently weights = scaling
    def get_scaled_accuracy_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        pred_one_hot = tf.one_hot(prediction, num_classes + 1) # go from label to one hot format to make binary easier
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        num_true = tf.count_nonzero(combined_label, axis=[1,2], keep_dims=True) # how many pixels of each class [1,2 are width/height]
        total_true = tf.reduce_sum(num_true, axis=[1,2,3]) # total true, calculate per batch, so wipe out all other axes
        scaling = 1 - num_true/total_true
                
        acc_str = 'xacc_mean_' + modality_str
        accuracy, _ = tf.metrics.accuracy(combined_label, pred_one_hot, weights=scaling)
        accuracy_loss = tf.reduce_mean(1 - accuracy, name=acc_str)
        tf.add_to_collection('losses', accuracy_loss)
        tf.summary.scalar(acc_str, accuracy_loss)
        return accuracy_loss, probabilities, prediction
    
        
    
    
    def get_precision_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        pred_one_hot = tf.one_hot(prediction, num_classes + 1) # go from label to one hot format to make binary easier
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        prec_str = 'xprec_mean_' + modality_str
        precision, _ = tf.metrics.precision(combined_label, pred_one_hot)
        precision_loss = tf.reduce_mean(1 - precision, name=prec_str) 
        tf.add_to_collection('losses', precision_loss)
        tf.summary.scalar(prec_str, precision_loss)
        return precision_loss, probabilities, prediction
    
    
    ###  may need to do precision individually for each class and then scale, currently weights = scaling
    def get_scaled_precision_loss(self, logits, modality_gt, bg_gt, num_classes, modality_str):
        probabilities = tf.nn.softmax(logits) # not flattened!
        prediction = tf.argmax(probabilities, axis=3) # which axis has the maximum value
        pred_one_hot = tf.one_hot(prediction, num_classes + 1) # go from label to one hot format to make binary easier
        
        smooth = 1e-5
        axis=[1,2,3]
        combined_label = tf.concat(values=[bg_gt, modality_gt], axis=3)
        
        num_true = tf.count_nonzero(combined_label, axis=[1,2], keep_dims=True) # how many pixels of each class [1,2 are width/height]
        total_true = tf.reduce_sum(num_true, axis=[1,2,3]) # total true, calculate per batch, so wipe out all other axes
        scaling = 1 - num_true/total_true
        
        prec_str = 'xprec_mean_' + modality_str
        precision, _ = tf.metrics.precision(combined_label, pred_one_hot, weights=scaling)
        precision_loss = tf.reduce_mean(1 - precision, name=prec_str) 
        tf.add_to_collection('losses', precision_loss)
        tf.summary.scalar(prec_str, precision_loss)
        return precision_loss, probabilities, prediction
    
    
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
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Creates a summary that measure the mean of activations.
        Creates a summary that measure the std dev of activations.
        Creates a summary that measure the max of activations.
        Creates a summary that measure the min of activations.
        Args:
          x: Tensor
        Returns:
          nothing
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
        if self.design.output_style == STYLE_SPLIT:
            self.val_cost = self.ct_cost + self.pt_cost
        elif self.design.output_style == STYLE_SINGLE:
            self.val_cost = self.all_cost
            
        tf.summary.scalar('validation_cost', self.val_cost)
        self.val_op = self.val_cost
      
    def _build_train_op(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for BN
        with tf.control_dependencies(update_ops): # for BN
            """Build training specific ops for the graph."""
            self.lrn_rate = tf.train.exponential_decay(self.hyper.lrn_rate, 
                                                       self.global_step,
                                                       self.hyper.lrn_rate_decay_steps,
                                                       self.hyper.lrn_rate_decay_rate, 
                                                       staircase=True, name='learning_rate')
            tf.summary.scalar('learning_rate', self.lrn_rate)

            trainable_variables = tf.trainable_variables()
        #grads = tf.gradients(self.cost, trainable_variables)
        #self.grads = grads

            if self.hyper.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hyper.optimizer == 'mom':
                optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

            grads = optimizer.compute_gradients(self.cost, trainable_variables)
        
            self.grads = grads
#        grad_check = tf.check_numerics(grads, 'Gradients exploded') #### <--- will not work attempts to pack grads of diff shapes
 #       with tf.control_dependencies([grad_check]):
            apply_op = optimizer.apply_gradients(
                grads,
            #zip(grads, trainable_variables),
                global_step=self.global_step, name='train_step')
        #print('%d groups of trainable variables' % len(trainable_variables))
        #print('%d groups of grads' % len(grads))
        #print(apply_op)

            train_ops = [apply_op] + self._extra_train_ops
        #print(train_ops)
            self.train_op = tf.group(*train_ops)
        #self.train_op = train_ops
        #print(self.train_op)
        
    
    def build_cross_modal_model(self):
        self.global_step = tf.train.get_or_create_global_step()
        
        width = self.design.width
        height = self.design.height
        depth = self.design.depth
        num_mods = self.design.num_mods
        fuse_depth = depth * num_mods
        
        input_ct = self.ct
        input_pet = self.pt
        input_fuse = self.fused
        
        self.construct_network(input_ct,input_pet,input_fuse)
        
        self._build_train_op()
        self._build_valid_op()
        self.summaries = tf.summary.merge_all()
        
        return
    
        
def main(unused_argv):
    print('RUNNING MAIN')
    default_design = DParams(width = 256, height = 256, depth = 1, num_mods = 2, num_filters = 64, conv_kern = [3,3],
                            pool_sz = [2,2], pool_str = 2, cross_mod_kern = [3,1,1], deconv_str = 2, cost='cross_entropy')
    default_hyper = HParams(batch_size=128,
                             num_classes=5,
                             num_epochs=1,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             lrn_rate_decay_steps=1000,
                             lrn_rate_decay_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom',
                             l2_scale=0.1,
                             loss_params=None)
    
    
    default_ct = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_ct')
    default_pt = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_pet')
    
    default_ctlb = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,3], name='default_ctlb')
    default_ptlb = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_ptlb')
    default_bglb = tf.placeholder(tf.float32, [None,default_design.width,default_design.height,1], name='default_bglb')
    
    net = CrossModalNet(default_hyper,default_design,default_ct,default_pt,default_ctlb,default_ptlb,default_bglb,'train')
    net.build_cross_modal_model()

    net.count_trainable_params()
    
    
    # need to do following in training due to presence of batch norm
    #    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #    with tf.control_dependencies(extra_update_ops):
    #        train_op = optimizer.minimize(loss)

if __name__ == "__main__":
    tf.app.run()