"""
    PET-CT Co-Learning Train/Eval module.
"""
import time
import six
import sys
import glob
import os

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.data import Iterator

import colearn_cnn as colearn_cnn
import colearn_input.petct_input as petct_input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'train (validation optional) or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'File path for training data.')
tf.app.flags.DEFINE_string('valid_data_path', '',
                           'File path for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'File path for eval data')

tf.app.flags.DEFINE_integer('image_size', 256, 'Image side length.')

tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('valid_dir', '',
                           'Directory to keep validation outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep evaluation outputs.')

tf.app.flags.DEFINE_integer('checkpoint_to_eval', 1000,
                           'Training iteration to use for evaluation')

tf.app.flags.DEFINE_integer('train_iter', 20,
                            'Number of iterations after which to log summaries.')
tf.app.flags.DEFINE_integer('chkpt_iter', 1000,
                            'Number of iterations after which to save checkpoint.')
tf.app.flags.DEFINE_integer('valid_iter', 20,
                            'Number of iterations after which to validate.')
tf.app.flags.DEFINE_integer('train_img_save_step', 2500,
                         'Whether to save training outputs as images (how many steps)')
tf.app.flags.DEFINE_integer('valid_img_save_step', 2500,
                         'Whether to save validation outputs as images (how many steps)')
tf.app.flags.DEFINE_integer('eval_img_save_step', 1,
                         'Whether to save evaluation outputs as images (how many steps)')

tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir, FLAGS.valid_dir, and FLAGS.eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_string('chkpt_file', 'model-',
                            'Checkpoint file name')

# hyper params
tf.app.flags.DEFINE_float('lrn_rate', 0.0001,
                         'Initial learning rate')
tf.app.flags.DEFINE_float('lrn_rate_decay_steps', 10000,
                         'Number of iterations after which to reduce the learning rate')
tf.app.flags.DEFINE_float('lrn_rate_decay_rate', 1.0,
                         'The factor by which the learning rate will be reduced')
tf.app.flags.DEFINE_float('relu_leakiness', 0.1,
                         'How much of -ve activations the leaky relu should let through')
tf.app.flags.DEFINE_float('l2_scale', 0.1,
                         'L2 Regularizer scale')
tf.app.flags.DEFINE_float('weight_decay_rate',0.0002,
                         'Weight decay rate')
tf.app.flags.DEFINE_string('optimizer','mom',
                           'Optimizer to use.')
tf.app.flags.DEFINE_integer('batch_size', 5,
                            'Batch size for training or eval')
tf.app.flags.DEFINE_integer('num_epochs', 500, 
                            'Number of epochs.')
tf.app.flags.DEFINE_integer('num_classes', 4, 
                            'Number of classes.')
tf.app.flags.DEFINE_integer('eval_epochs', 1, 
                            'Number of epochs to evaluate test data (mainly to check consistency).')


def _get_tfrecord_files_from_dir(the_dir):
    TFREC_FILTER = os.path.join(the_dir,'*.tfrecords')
    return glob.glob(TFREC_FILTER)


# this can probably be refactored but not critial for now
def _saveImages(num_images, step, cts, pts, labels, probs, mode='train'):
    if mode == 'train':
        save_dir = FLAGS.train_dir
    elif mode == 'valid':
        save_dir = FLAGS.valid_dir
    elif mode == 'eval':
        save_dir = FLAGS.eval_dir
            
    for i in range(num_images):
        ct_data = cts[i,:,:,:] # [4 dims because batch, width, height, channels]
        ct_data = np.moveaxis(ct_data, -1, 0)
        ct_list = []
        for sl in ct_data:
            ct_list.append(Image.fromarray(sl,mode='F'))
        ct_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_ct.tif',
                        save_all=True)
        
        pt_data = pts[i,:,:,:] # [4 dims because batch, width, height, depth]
        pt_data = np.moveaxis(pt_data, -1, 0)
        pt_list = []
        for sl in pt_data:
            pt_list.append(Image.fromarray(sl,mode='F'))
        pt_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_pt.tif',
                        save_all=True)
        
        lbl_data = labels[i,:,:,:] # [4 dims because batch, width, height, depth]
        lbl_data = np.moveaxis(lbl_data, -1, 0)
        lbl_list = []
        for sl in lbl_data:
            lbl_list.append(Image.fromarray(sl,mode='F'))
        lbl_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_truth.tif',
                        save_all=True,
                        append_images=lbl_list[1:])

        if probs is not None:
            prob_data = probs[i,:,:,:] # [4 dims because batch, width, height, depth]
            prob_data = np.moveaxis(prob_data, -1, 0)
            prob_list = []
            for sl in prob_data:
                prob_list.append(Image.fromarray(sl,mode='F'))
            prob_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_prob.tif',
                            save_all=True,
                            append_images=prob_list[1:])
            

            
def _saveFusionMap(num_images, num_blocks, step, colearn_outputs, mode='eval'):
    if mode == 'train':
        save_dir = FLAGS.train_dir
    elif mode == 'valid':
        save_dir = FLAGS.valid_dir
    elif mode == 'eval':
        save_dir = FLAGS.eval_dir
        
    if colearn_outputs is not None:
        for blk in range(num_blocks): # each element represents a block
            fusion_map = colearn_outputs[blk]
            for i in range(num_images):
                fusion_data = fusion_map[i,:,:,:] # [4 dims because batch, width, height, depth]
                fusion_data = np.moveaxis(fusion_data, -1, 0)
                fusion_list = []
                for sl in fusion_data:
                    fusion_list.append(Image.fromarray(sl,mode='F'))
                fusion_list[0].save(save_dir + '/step' + str(step) + '_batch_' + str(i) + '_blk_' + str(blk) + '_fusion.tif',
                                save_all=True,
                                append_images=fusion_list[1:])
            

            
            
def get_metrics_ops(model, mode='train'):
    truth = model.label
    probs = model.probabilities
    pred_slice = tf.argmax(probs, axis=3) # which axis has the maximum value    
    predictions = tf.one_hot(pred_slice, FLAGS.num_classes) # go from label to one hot format to make binary easier
    
    _, precision_op = tf.metrics.precision(truth, predictions, name=mode+'_precision') # ignore the ops output
    _, recall_op = tf.metrics.recall(truth, predictions, name=mode+'_recall') # ignore the ops output
    _, accuracy_op = tf.metrics.accuracy(truth, predictions, name=mode+'_accuracy') # ignore the ops output
    _, rmse_op = tf.metrics.root_mean_squared_error(truth, predictions, name=mode+'_rmse') # ignore the ops output
    
    if mode=='train':
        summary_op = tf.summary.merge([model.summaries,
                tf.summary.scalar('Precision', precision_op),
                tf.summary.scalar('Recall', recall_op),
                tf.summary.scalar('Accuracy', accuracy_op),
                tf.summary.scalar('RMSE', rmse_op)], name=mode+'_summary')       
    else:
        summary_op = tf.summary.merge([tf.summary.scalar('Loss', model.loss),
                tf.summary.scalar('Precision', precision_op),
                tf.summary.scalar('Recall', recall_op),
                tf.summary.scalar('Accuracy', accuracy_op),
                tf.summary.scalar('RMSE', rmse_op)], name=mode+'_summary')
    return summary_op, precision_op, recall_op, accuracy_op, rmse_op       
    
    
    
def train(hps, design):
    """Training loop."""
    # use a placeholder for batch size, to allow feeding for eval
    batch_size = tf.placeholder(tf.int64, shape=[], name='batch')
    
    # build training input iterator
    train_records = _get_tfrecord_files_from_dir(FLAGS.train_data_path) #get tfrecord files for train
    train_dataset, modality_names = petct_input.build_input(train_records, batch_size, hps.num_epochs, design.width, train_mode=True)
    train_iterator = train_dataset.make_initializable_iterator()    
    
    # build validation input iterator - skip validation if no path
    if not FLAGS.valid_data_path == '':        
        val_records = _get_tfrecord_files_from_dir(FLAGS.valid_data_path) # get tfrecord files for val
        val_dataset, _ = petct_input.build_input(val_records, batch_size, hps.num_epochs, design.width, train_mode=False)
        val_iterator = val_dataset.make_initializable_iterator()
    
    # use placeholder handle to get data from input
    handle = tf.placeholder(tf.string, shape=[], name='data')
    iterator = Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    modalities, labels = iterator.get_next()
        
    # build architecture
    model = colearn_cnn.CoLearnNet(hps, design, modalities, modality_names, labels) 
    model.build_colearn_architecture()
        
    # get metrics ops here for train and val 
    with tf.variable_scope('metrics'):
        tr_summary_op, tr_precision_op, tr_recall_op, tr_accuracy_op, tr_rmse_op = get_metrics_ops(model,'train')
        val_summary_op, val_precision_op, val_recall_op, val_accuracy_op, val_rmse_op = get_metrics_ops(model,'valid')
      
    # needed for input handlers
    g_init_op = tf.global_variables_initializer()
    l_init_op = tf.local_variables_initializer()


    # sessuon runs on GPU          
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count = {'GPU': 1})) as mon_sess: 
        # Need a saver to save and restore all the variables.
        saver = tf.train.Saver()
                
        # define handles and summary writers
        training_handle = mon_sess.run(train_iterator.string_handle())
        train_writer = tf.summary.FileWriter(FLAGS.log_root + '/train',
                                         mon_sess.graph)
        mon_sess.run(train_iterator.initializer, feed_dict={batch_size: hps.batch_size})
        
        # skip validation if no path
        if not FLAGS.valid_data_path == '':        
            validation_handle = mon_sess.run(val_iterator.string_handle())
            valid_writer = tf.summary.FileWriter(FLAGS.log_root + '/valid')
            mon_sess.run(val_iterator.initializer, feed_dict={batch_size: hps.batch_size})
        
        # run initializers
        mon_sess.run([g_init_op, l_init_op])
        
        train_summary = None
        step = None
        val_summary = None        

        while True:
            try:
                ## run training op
                _, train_summary, step, loss, p, r, a, e, cts, pts, lbls, probs = mon_sess.run([model.train_op, tr_summary_op, model.global_step, model.loss, tr_precision_op, tr_recall_op, tr_accuracy_op, tr_rmse_op, model.data[0], model.data[1], model.label, model.probabilities], feed_dict={handle: training_handle, batch_size: hps.batch_size, model.is_training: True})
                    
                # check if there is a need to print train logs
                if FLAGS.train_iter > 0 and step % FLAGS.train_iter == 0:
                    print('[TRAIN] STEP: %d, LOSS: %.5f, PRECISION: %.5f, RECALL: %.5f, ACCURACY: %.5f, RMSE: %.5f' %
                              (step, loss, p, r, a, e))
                    train_writer.add_summary(train_summary, step)
                    train_writer.flush()
            
                    # print images if needed
                    if FLAGS.train_img_save_step > 0 and step % FLAGS.train_img_save_step == 0:
                        print('SAVING TRAINING IMAGES')
                        _saveImages(hps.batch_size, step, cts, pts, labels = lbls, probs = probs, mode='train')
                        
                # run validation op AND print validation logs if specified
                if not FLAGS.valid_data_path == '' and FLAGS.valid_iter > 0 and step % FLAGS.valid_iter == 0:
                    _, val_summary, loss, p, r, a, e, cts, pts, lbls, probs = mon_sess.run([model.val_op, val_summary_op, model.loss, val_precision_op, val_recall_op, val_accuracy_op, val_rmse_op, model.data[0], model.data[1], model.label, model.probabilities], feed_dict={handle: validation_handle, batch_size: hps.batch_size, model.is_training: False})
                    val_step = step
                    print('[VALID] STEP: %d, LOSS: %.5f, PRECISION: %.5f, RECALL: %.5f, ACCURACY: %.5f, RMSE: %.5f' % 
                              (step, loss, p, r, a, e))
                    valid_writer.add_summary(val_summary, step)
                    valid_writer.flush()
                        
                    if FLAGS.valid_img_save_step > 0 and step % FLAGS.valid_img_save_step == 0:
                        print('SAVING VALIDATION IMAGES')
                        _saveImages(hps.batch_size, step, cts, pts, labels = lbls, probs = probs, mode='valid')
                
                # save checkpoint when specified
                if FLAGS.chkpt_iter > 0 and step % FLAGS.chkpt_iter == 0:
                    save_loc = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(step) + '.ckpt'
                    save_path = saver.save(mon_sess, save_loc)
                    print('Model saved in path: %s' % save_path)
            
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                # now finished training (either train or validation has run out)
                train_writer.add_summary(train_summary, step)
                train_writer.flush()
                # skip validation if no path
                if not FLAGS.valid_data_path == '':        
                    valid_writer.add_summary(val_summary, val_step)
                    valid_writer.flush()
                save_loc = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(step) + '-end.ckpt'
                save_path = saver.save(mon_sess, save_loc)
                print('Model saved in path: %s' % save_path)
                break
            
            

            
def evaluate(hps, design):
    """Eval loop."""
    # define eval input data
    eval_records = _get_tfrecord_files_from_dir(FLAGS.eval_data_path) #get tfrecord files for train
    eval_dataset, _ = petct_input.build_input(eval_records, hps.batch_size, FLAGS.eval_epochs, design.width, train_mode=False) 
    eval_iterator = eval_dataset.make_initializable_iterator()
    
    # needed for input handlers
    g_init_op = tf.global_variables_initializer()
    l_init_op = tf.local_variables_initializer()
    
    # make directory where outputs will be stored
    if FLAGS.eval_dir != '' and not os.path.exists(FLAGS.eval_dir):
        os.makedirs(FLAGS.eval_dir)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count = {'GPU': 1})) as mon_sess:
        # initialise
        mon_sess.run([g_init_op, l_init_op])
       
        # get graph definition
        meta_graph_file = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(FLAGS.checkpoint_to_eval) + '-end.ckpt.meta'        
        
        # get checkpoint
        ckpt_file = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(FLAGS.checkpoint_to_eval) + '-end.ckpt'
        
        # try to load
        try:
            tf.logging.info('Loading checkpoint %s', ckpt_file)
            meta_restore = tf.train.import_meta_graph(meta_graph_file)        
            meta_restore.restore(mon_sess, ckpt_file)
        except tf.errors.OpError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            sys.exit(0)
        
        # get all the tensors and operations that need to be fed during evaluation
        handle = tf.get_default_graph().get_tensor_by_name('data:0') # data will be fed here 
        batch_size = tf.get_default_graph().get_tensor_by_name('batch_size:0') # batch_size variation
        train_mode  = tf.get_default_graph().get_tensor_by_name('train_mode:0') # will be set to False to turn off batch norm 
        
        # iterator initlisation after we have gotten the batch size placeholder from the checkpoint
        eval_handle = mon_sess.run(eval_iterator.string_handle())
        mon_sess.run(eval_iterator.initializer, feed_dict={batch_size: hps.batch_size})
        
        # get all the tensors and operations that need to be monitored during evaluation
        probabilities = tf.get_default_graph().get_tensor_by_name('probability_map:0') 
        ct_img = tf.get_default_graph().get_tensor_by_name('CT:0') 
        pt_img = tf.get_default_graph().get_tensor_by_name('PET:0') 
        label = tf.get_default_graph().get_tensor_by_name('label:0') 
        colearn_ops = []
        for blk in range(1, design.num_blocks + 1):
            colearn_op = tf.get_default_graph().get_tensor_by_name('COLEARN_' + str(blk) + '/Squeeze:0') 
            colearn_ops.append(colearn_op)
        
        step = 0
        while True:
            try:
                step = step + 1
            
                # forward pass
                cts, pts, lbls, probs, colearn_out = mon_sess.run([ct_img, pt_img, label, probabilities, colearn_ops], feed_dict={handle: eval_handle, batch_size: hps.batch_size, train_mode: False})

                print('[EVAL] STEP: %d' % (step))
                if FLAGS.eval_img_save_step > 0:
                    if step % FLAGS.eval_img_save_step == 0:
                        # only works for single style
                        print('SAVING EVAL IMAGES')
                        _saveImages(cts.shape[0], step, cts, pts, labels = lbls, probs = probs, mode='eval')
                        print('SAVING CO-LEARNED FUSION')
                        _saveFusionMap(cts.shape[0], design.num_blocks, step, colearn_out, mode='eval')

            #out of data
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                break



def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    # hyper params for colearning CNN
    hps = colearn_cnn.HParams(batch_size=FLAGS.batch_size,
                             num_classes=FLAGS.num_classes,
                             num_epochs=FLAGS.num_epochs,
                             lrn_rate=FLAGS.lrn_rate,
                             lrn_rate_decay_steps=FLAGS.lrn_rate_decay_steps,
                             lrn_rate_decay_rate=FLAGS.lrn_rate_decay_rate,
                             weight_decay_rate=FLAGS.weight_decay_rate,
                             relu_leakiness=FLAGS.relu_leakiness,
                             optimizer=FLAGS.optimizer,
                             l2_scale=FLAGS.l2_scale)
    
    # design parameters for colearning CNN
    design = colearn_cnn.DParams(width = 256, height = 256, depth = 1, num_mods = 2, num_filters = 64, conv_kernel = [3,3],
                            pool_sz = [2,2], pool_str = 2, colearn_kernel = [2,3,3], deconv_str = 1, num_blocks = 4)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps, design)
        elif FLAGS.mode == 'eval':
            evaluate(hps, design)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()