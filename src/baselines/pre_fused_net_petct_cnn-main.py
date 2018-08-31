"""Cross Modality PET-CT Co-Learn Train/Eval module.
"""
import time
import six
import sys
import glob
import os

import petct_input
import numpy as np
import scipy as sp
import pre_fused_petct_cnn
import tensorflow as tf

from PIL import Image
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.data import Iterator


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'petct', 'MM data.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('val_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')

tf.app.flags.DEFINE_integer('image_size', 256, 'Image side length.')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')

tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')

tf.app.flags.DEFINE_integer('eval_chkpt_num', 1000,
                           'Checkpoint to use for evaluation')

tf.app.flags.DEFINE_integer('train_iter', 10,
                            'Number of iterations after which to log summaries.')
tf.app.flags.DEFINE_integer('chkpt_iter', 10,
                            'Number of iterations after which to save checkpoint.')
tf.app.flags.DEFINE_integer('val_iter', 10,
                            'Number of iterations after which to validate.')

tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 5,
                            'Batch size for training or eval')
tf.app.flags.DEFINE_string('chkpt_file', 'model-',
                            'Checkpoint file name')


tf.app.flags.DEFINE_bool('DEBUG', False,
                         'Whether to run in a debug wrapper')
tf.app.flags.DEFINE_bool('RESUME', False,
                         'Whether to resume a training run')

tf.app.flags.DEFINE_integer('IMSAVE', 0,
                         'Whether to save outputs as images (how many steps)')

# hyper params
tf.app.flags.DEFINE_float('lrn_rate', 0.1,
                         'Initial learning rate')
tf.app.flags.DEFINE_float('lrn_rate_decay_steps', 1000,
                         'Number of iterations after which to reduce the learning rate')
tf.app.flags.DEFINE_float('lrn_rate_decay_rate', 0.1,
                         'The factor by which the learning rate will be reduced')
tf.app.flags.DEFINE_float('relu_leakiness', 0.1,
                         'How much of -ve activations the leaky relu should let through')
tf.app.flags.DEFINE_float('l2_scale', 0.1,
                         'L2 Regularizer scale')
tf.app.flags.DEFINE_string('cost', pre_fused_petct_cnn.CROSS_ENTROPY,
                            'Cost function')
tf.app.flags.DEFINE_float('loss_params', None,
                            'Cost function parameters')
tf.app.flags.DEFINE_string('output_style', pre_fused_petct_cnn.STYLE_DEFAULT,
                            'output style')




def _get_tfrecord_files_from_dir(the_dir):
    TFREC_FILTER = os.path.join(the_dir,'*.tfrecords')
    return glob.glob(TFREC_FILTER)


def _saveImages(num_images, step, cts, pts, trcts=None, trpts=None, trallpos=None, trbgs=None, recon_cts=None, recon_pts=None, recon_all=None, ct_preds=None, pt_preds=None, all_preds=None):
    if trcts is not None:
        trs = np.concatenate([trbgs, trcts,trpts],3) # 4 dims so concat channels
    else:
        trs = np.concatenate([trbgs, trallpos],3)
    
    if FLAGS.mode == 'train':
        save_dir = FLAGS.train_dir
    else:
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
        
        tr_data = trs[i,:,:,:] # [4 dims because batch, width, height, depth]
        tr_data = np.moveaxis(tr_data, -1, 0)
        tr_list = []
        for sl in tr_data:
            tr_list.append(Image.fromarray(sl,mode='F'))
        tr_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_truth.tif',
                        save_all=True,
                        append_images=tr_list[1:])
        

        if recon_cts is not None:
            rec_ct_data = recon_cts[i,:,:] # [3 dims because batch, width, height]
            #rec_ct_data = np.moveaxis(rec_ct_data, -1, 0)
            img_array = np.stack((rec_ct_data,)*3, -1)
            img_array = np.asarray(img_array,dtype='uint8')
            rec_ct_img = Image.fromarray(img_array)
            rec_ct_img.save(save_dir + '/' + str(step) + '_' + str(i) + '_rec_ct.jpg')

        if recon_pts is not None:
            rec_pt_data = recon_pts[i,:,:] # [3 dims because batch, width, height]
            #rec_pt_data = np.moveaxis(rec_pt_data, -1, 0)
            img_array = np.stack((rec_pt_data,)*3, -1)
            img_array = np.asarray(img_array,dtype='uint8')
            rec_pt_img = Image.fromarray(img_array)
            rec_pt_img.save(save_dir + '/' + str(step) + '_' + str(i) + '_rec_pt.jpg')
            
        if recon_all is not None:
            rec_all_data = recon_all[i,:,:] # [3 dims because batch, width, height]
            #rec_all_data = np.moveaxis(rec_all_data, -1, 0)
            img_array = np.stack((rec_all_data,)*3, -1)
            img_array = np.asarray(img_array,dtype='uint8')
            rec_all_img = Image.fromarray(img_array)
            rec_all_img.save(save_dir + '/' + str(step) + '_' + str(i) + '_rec_all.jpg')

        if ct_preds is not None:
            ct_pred_data = ct_preds[i,:,:] # [4 dims because batch, width, height, depth]
            ct_pred_data = np.moveaxis(ct_pred_data, -1, 0)
            ct_pred_list = []
            for sl in ct_pred_data:
                ct_pred_list.append(Image.fromarray(sl,mode='F'))
            ct_pred_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_ct_pred.tif',
                            save_all=True,
                            append_images=ct_pred_list[1:])

        if pt_preds is not None:
            pt_pred_data = pt_preds[i,:,:] # [4 dims because batch, width, height, depth]
            pt_pred_data = np.moveaxis(pt_pred_data, -1, 0)
            pt_pred_list = []
            for sl in pt_pred_data:
                pt_pred_list.append(Image.fromarray(sl,mode='F'))
            pt_pred_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_pt_pred.tif',
                            save_all=True,
                            append_images=pt_pred_list[1:])

        if all_preds is not None:
            all_pred_data = all_preds[i,:,:] # [4 dims because batch, width, height, depth]
            all_pred_data = np.moveaxis(all_pred_data, -1, 0)
            all_pred_list = []
            for sl in all_pred_data:
                all_pred_list.append(Image.fromarray(sl,mode='F'))
            all_pred_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_all_pred.tif',
                            save_all=True,
                            append_images=all_pred_list[1:])
            
def get_metrics_ops(model, mode='train'):
    # modify below for GT and accuracy based on ROIs # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEED TO ADD BG???????????????????????   
    ## CHECK THE METRICS BELOW - MAY NEED TO FIX argmax/onehot to account for background  in the SPLIT! 
    if FLAGS.output_style == pre_fused_petct_cnn.STYLE_SPLIT:
        ct_shape = model.recon_ct.get_shape().as_list()
        pt_shape = model.recon_pt.get_shape().as_list()
        ct_shape[3] = ct_shape[3]-1
        pt_shape[3] = pt_shape[3]-1
        
        truth = tf.concat([model.lbct, model.lbpt], 3)
        pred_ct = model.ct_probabilities[:,:,:,1:] # first dimension should be background
        pred_pt = model.pt_probabilities[:,:,:,1:] # first dimension should be background
        probs = tf.concat([pred_ct, pred_pt], 3) # 3rd dim because [batch, height, width, depth]
        one_hot_depth = ct_shape[3] + pt_shape[3]
    elif FLAGS.output_style == pre_fused_petct_cnn.STYLE_SINGLE:
        truth = tf.concat([model.lbbg, model.lb_pos_gt], 3)
        probs = model.all_probabilities
        one_hot_depth = model.num_all_classes + 1
        
    pred_slice = tf.argmax(probs, axis=3) # which axis has the maximum value
    predictions = tf.one_hot(pred_slice, one_hot_depth) # go from label to one hot format to make binary easier
    _, precision_op = tf.metrics.precision(truth, predictions, name=mode+'_precision') # ignore the ops output
    _, recall_op = tf.metrics.recall(truth, predictions, name=mode+'_recall') # ignore the ops output
    _, accuracy_op = tf.metrics.accuracy(truth, predictions, name=mode+'_accuracy') # ignore the ops output
    _, rmse_op = tf.metrics.root_mean_squared_error(truth, predictions, name=mode+'_rmse') # ignore the ops output
    
    # due to error in batch norm, not using model .summaries for mode=='valid'
    if mode=='train':
        summary_op = tf.summary.merge([model.summaries,
                tf.summary.scalar('Precision', precision_op),
                tf.summary.scalar('Recall', recall_op),
                tf.summary.scalar('Accuracy', accuracy_op),
                tf.summary.scalar('RMSE', rmse_op)], name=mode+'_summary')
    else:
        summary_op = tf.summary.merge([#model.summaries,
                tf.summary.scalar('Precision', precision_op),
                tf.summary.scalar('Recall', recall_op),
                tf.summary.scalar('Accuracy', accuracy_op),
                tf.summary.scalar('RMSE', rmse_op)], name=mode+'_summary')
        
    return summary_op, precision_op, recall_op, accuracy_op, rmse_op
        

def train(hps, design):
    """Training loop."""
    train_records = _get_tfrecord_files_from_dir(FLAGS.train_data_path) #get tfrecord files for train
    train_iterator = petct_input.build_input(train_records,
                                             hps.batch_size, hps.num_epochs, FLAGS.mode)
    train_iterator_handle = train_iterator.string_handle()
    
    if not FLAGS.val_data_path == '':        # skip validation if no path
        val_records = _get_tfrecord_files_from_dir(FLAGS.val_data_path) # get tfrecord files for val
        val_iterator = petct_input.build_input(val_records,
                                           hps.batch_size, hps.num_epochs, 'valid')
        val_iterator_handle = val_iterator.string_handle()
    
    handle = tf.placeholder(tf.string, shape=[], name='data')
    iterator = Iterator.from_string_handle(handle, train_iterator.output_types, train_iterator.output_shapes)
    ct, pt, ctlb, ptlb, bglb = iterator.get_next()
    
    
    model = pre_fused_petct_cnn.PreFusedNet(hps, design, ct, pt, ctlb, ptlb, bglb, FLAGS.mode) 
    model.build_cross_modal_model()
    
    # for use in loading later
    #tf.get_collection('model')
    #tf.add_to_collection('model',model)
    
    
    # put get metrics ops here for train and val 
    with tf.variable_scope('metrics'):
        tr_summary_op, tr_precision_op, tr_recall_op, tr_accuracy_op, tr_rmse_op = get_metrics_ops(model,'train')
        val_summary_op, val_precision_op, val_recall_op, val_accuracy_op, val_rmse_op = get_metrics_ops(model,'valid')
    
  
    # needed for input handlers
    g_init_op = tf.global_variables_initializer()
    l_init_op = tf.local_variables_initializer()


              
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count = {'GPU': 1})) as mon_sess: 
        # Need a saver to save and restore all the variables.
        saver = tf.train.Saver()
        
        if FLAGS.DEBUG:
            print('ENABLING DEBUG')
            mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess)
            mon_sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        
        training_handle = mon_sess.run(train_iterator_handle)
        if not FLAGS.val_data_path == '':        # skip validation if no path
            validation_handle = mon_sess.run(val_iterator_handle)
        
        train_writer = tf.summary.FileWriter(FLAGS.log_root + '/train',
                                         mon_sess.graph)
        
        if not FLAGS.val_data_path == '':        # skip validation if no path
            valid_writer = tf.summary.FileWriter(FLAGS.log_root + '/valid')
        
        mon_sess.run([g_init_op, l_init_op])
        
        
        summary = None
        step = None
        val_summary = None
        #check = 1
        while True:
            try:
                ## FIRST RUN TRAINING OP BASED ON OUTPUT STYLE
                if FLAGS.output_style == pre_fused_petct_cnn.STYLE_SPLIT:
                    # get PET and CT recons separately
                    _, summary, step, loss, p, r, a, e, cts, pts, trcts, trpts, trbgs, recon_cts, recon_pts, ct_preds, pt_preds = mon_sess.run([model.train_op, tr_summary_op, model.global_step, model.cost, tr_precision_op, tr_recall_op, tr_accuracy_op, tr_rmse_op, model.ct, model.pt, model.lbct, model.lbpt, model.lbbg, model.ct_pred, model.pt_pred, model.ct_probabilities, model.pt_probabilities], feed_dict={handle: training_handle, model.is_training: True})
                elif FLAGS.output_style == pre_fused_petct_cnn.STYLE_SINGLE:
                    # get PET and CT recons together
                    _, summary, step, loss, p, r, a, e, cts, pts, trallpos, trbgs, recon_all, all_preds = mon_sess.run([model.train_op, tr_summary_op, model.global_step, model.cost, tr_precision_op, tr_recall_op, tr_accuracy_op, tr_rmse_op, model.ct, model.pt, model.lb_pos_gt, model.lbbg, model.all_pred, model.all_probabilities], feed_dict={handle: training_handle, model.is_training: True})
                    
                    
                    
                if step % FLAGS.train_iter == 0:
                    print('[TRAIN] STEP: %d, LOSS: %.5f, PRECISION: %.5f, RECALL: %.5f, ACCURACY: %.5f, RMSE: %.5f' % (step, loss, p, r, a, e))
                    train_writer.add_summary(summary, step)
                    train_writer.flush()
            
                if FLAGS.IMSAVE > 0:
                    if step % FLAGS.IMSAVE == 0:
                        print('SAVING IMAGES')
                        if FLAGS.output_style == pre_fused_petct_cnn.STYLE_SPLIT:
                            _saveImages(hps.batch_size, step, cts, pts, trcts=trcts, trpts=trpts, trbgs=trbgs, recon_cts=recon_cts, recon_pts=recon_pts, ct_preds=ct_preds, pt_preds=pt_preds)
                        elif FLAGS.output_style == pre_fused_petct_cnn.STYLE_SINGLE:
                            _saveImages(hps.batch_size, step, cts, pts, trallpos=trallpos, trbgs=trbgs, recon_all=recon_all, all_preds = all_preds)
                        
            
                if not FLAGS.val_data_path == '':        # skip validation if no path
                    if step % FLAGS.val_iter == 0:
                        _, val_summary, loss, p, r, a, e = mon_sess.run([model.val_op, val_summary_op, model.cost, val_precision_op, val_recall_op, val_accuracy_op, val_rmse_op], feed_dict={handle: validation_handle, model.is_training: False})
                        val_step = step
                        print('[VALID] STEP: %d, LOSS: %.5f, PRECISION: %.5f, RECALL: %.5f, ACCURACY: %.5f, RMSE: %.5f' % (step, loss, p, r, a, e))
                        valid_writer.add_summary(val_summary, step)
                        valid_writer.flush()
                
                if step % FLAGS.chkpt_iter == 0:
                    save_loc = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(step) + '.ckpt'
                    save_path = saver.save(mon_sess, save_loc)
                    print('Model saved in path: %s' % save_path)
            
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                # now finished training (either train or validation has run out)
                train_writer.add_summary(summary, step)
                train_writer.flush()
                if not FLAGS.val_data_path == '':        # skip validation if no path
                    valid_writer.add_summary(val_summary, val_step)
                    valid_writer.flush()
                save_loc = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(step) + '-end.ckpt'
                save_path = saver.save(mon_sess, save_loc)
                print('Model saved in path: %s' % save_path)
                break
            
            

            
def evaluate(hps, design):
    """Eval loop."""
    
    eval_records = _get_tfrecord_files_from_dir(FLAGS.eval_data_path) #get tfrecord files for train
    eval_iterator = petct_input.build_input(eval_records,
                                             hps.batch_size, hps.num_epochs, FLAGS.mode) 
    eval_iterator_handle = eval_iterator.string_handle()
    
    #handle = tf.placeholder(tf.string, shape=[], name='data')
    #iterator = Iterator.from_string_handle(handle, eval_iterator.output_types, eval_iterator.output_shapes)
    #ct, pt, ctlb, ptlb, bglb = iterator.get_next()
    
    
    #model = pre_fused_petct_cnn.PreFusedNet(hps, design, ct, pt, ctlb, ptlb, bglb, FLAGS.mode) 
    #model.build_cross_modal_model()
    # put get metrics ops here for train and val 
    #eval_summary_op, eval_precision_op, eval_recall_op, eval_accuracy_op, eval_rmse_op = get_metrics_ops(model)
    
    # needed for input handlers
    g_init_op = tf.global_variables_initializer()
    l_init_op = tf.local_variables_initializer()
    

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count = {'GPU': 1})) as mon_sess:
        mon_sess.run([g_init_op, l_init_op])
        
        ckpt_meta = FLAGS.log_root + '/' + FLAGS.chkpt_file + str(FLAGS.eval_chkpt_num) + '-end.ckpt.meta'
        meta_restore = tf.train.import_meta_graph(ckpt_meta)
    
        #ckpt_saver = tf.train.Saver()
        eval_writer = tf.summary.FileWriter(FLAGS.eval_dir)
       
        eval_handle = mon_sess.run(eval_iterator_handle)               
        
        
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            sys.exit(0)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            sys.exit(0)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        #ckpt_saver.restore(mon_sess, ckpt_state.model_checkpoint_path) 
        meta_restore.restore(mon_sess, ckpt_state.model_checkpoint_path) 
        
        # get all the tensors and operations that need to be fed during evaluation
        handle = tf.get_default_graph().get_tensor_by_name('data:0') # data will be fed here #works
        train_mode  = tf.get_default_graph().get_tensor_by_name('train_mode:0') # will be set to False to turn off BN #works
        
        # get all the tensors and operations that need to be monitored during evaluation
        # needed to get metrics
        #eval_summary_op = tf.get_default_graph().get_tensor_by_name('metrics/valid_summary/valid_summary:0')#works
        #eval_precision_op = tf.get_default_graph().get_tensor_by_name('metrics/valid_precision/update_op:0')#works
        #eval_recall_op = tf.get_default_graph().get_tensor_by_name('metrics/valid_recall/update_op:0')#works
        #eval_accuracy_op = tf.get_default_graph().get_tensor_by_name('metrics/valid_accuracy/update_op:0')#works
        #eval_rmse_op = tf.get_default_graph().get_tensor_by_name('metrics/Sqrt_3:0')#works BUT not init (maybe ignore?)
        all_probabilities = tf.get_default_graph().get_tensor_by_name('costs/all_probabilities:0') #works
        all_pred = tf.get_default_graph().get_tensor_by_name('costs/all_prediction:0') #works
        ct_img = tf.get_default_graph().get_tensor_by_name('ct:0')#works
        pt_img = tf.get_default_graph().get_tensor_by_name('pt:0')#works
        lb_pos_gt = tf.get_default_graph().get_tensor_by_name('lb_pos_gt:0') #works
        lbbg = tf.get_default_graph().get_tensor_by_name('lbbg:0')#works
        
        
        
    
        step = 0
        while True:
        
            try:
                step = step + 1
                # modify below! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                cts, pts, trallpos, trbgs, recon_all, all_preds = mon_sess.run([ct_img, pt_img, lb_pos_gt, lbbg, all_pred, all_probabilities], feed_dict={handle: eval_handle, train_mode: False})
                #eval_summary, p, r, a, e, cts, pts, trallpos, trbgs, recon_all, all_preds = mon_sess.run([eval_summary_op, eval_precision_op, eval_recall_op, eval_accuracy_op, eval_rmse_op, ct_img, pt_img, lb_pos_gt, lbbg, all_pred, all_probabilities], feed_dict={handle: eval_handle, train_mode: False})
                print('[EVAL] STEP: %d' % (step))
                #print('[EVAL] STEP: %d, PRECISION: %.5f, RECALL: %.5f, ACCURACY: %.5f, RMSE: %.5f' % (step, p, r, a, e))
                #eval_writer.add_summary(eval_summary, step)
                #eval_writer.flush()
                
                if FLAGS.IMSAVE > 0:
                    if step % FLAGS.IMSAVE == 0:
                        # only works for single style
                        print('SAVING IMAGES')
                        _saveImages(hps.batch_size, step, cts, pts, trallpos=trallpos, trbgs=trbgs, recon_all=recon_all, all_preds = all_preds)


            

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

    batch_size = FLAGS.batch_size
    #if FLAGS.mode == 'train':
    #    batch_size = 128
    #elif FLAGS.mode == 'eval':
    #    batch_size = 100
        
    num_epochs = FLAGS.num_epochs
    
    cost = FLAGS.cost
    loss_params = FLAGS.loss_params
    
    style = FLAGS.output_style

    # this needs to be changed based on FCN classes setting
    num_classes = 5

    # change the below to be cross-mod-conv params and not resnet
    hps = pre_fused_petct_cnn.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             num_epochs=num_epochs,
                             min_lrn_rate=0.0001,
                             lrn_rate=FLAGS.lrn_rate,
                             lrn_rate_decay_steps=FLAGS.lrn_rate_decay_steps,
                             lrn_rate_decay_rate=FLAGS.lrn_rate_decay_rate,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=FLAGS.relu_leakiness,
                             optimizer='mom',
                             l2_scale=FLAGS.l2_scale,
                             loss_params = loss_params)
    design = pre_fused_petct_cnn.DParams(width = 256, height = 256, depth = 1, num_mods = 2, num_filters = 64, conv_kern = [3,3],
                            pool_sz = [2,2], pool_str = 2, cross_mod_kern = [2,3,3], deconv_str = 2, 
                            cost=cost, output_style=style)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps, design)
        elif FLAGS.mode == 'eval':
            print('EVALUATION MODE')
            evaluate(hps, design)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()