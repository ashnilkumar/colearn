import sys
import glob
import os
import numpy as np
import scipy as sp
import tensorflow as tf

from PIL import Image

def _get_tfrecord_files_from_dir(the_dir,flag='all'):
    TFREC_FILTER = os.path.join(the_dir,'*'+flag+'.tfrecords')
    return glob.glob(TFREC_FILTER)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
def precision(y_true, y_pred):
    #y_pred_num = K.argmax(y_pred, axis=-1)
    #true_positives = K.sum(K.cast(y_true == y_pred_num, K.floatx()))
    #y_true = y_true[:,0:3]
    #y_pred = y_pred[:,0:3]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #true_positives = K.sum(y_true == y_pred)
    predicted_positives = K.sum(y_pred)
    #predicted_positives = K.sum(y_pred)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def recall(y_true, y_pred):
    #y_pred_num = K.argmax(y_pred, axis=-1)
    #true_positives = K.sum(K.cast(y_true == y_pred_num, K.floatx()))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #true_positives = K.sum(y_true == y_pred)
    possible_positives = K.sum(y_true)
    #possible_positives = K.sum(y_true, 0, 1)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def serving_input_fn():
    feature_spec = {'input_ct': tf.FixedLenFeature(dtype=tf.float32, shape=[256,256,1]), 'input_cp': tf.FixedLenFeature(dtype=tf.float32, shape=[256,256,1])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def modality_loss(y_true, y_pred):
    loss = K.sum(K.abs(y_pred - y_true), axis=-1)
    tf.Print('loss: ', [loss])
    return loss


def _saveImages(num_images, step, gt, probs):
    save_dir = 'baseline_vantulder_convae/outputs'
    for i in range(num_images):
        tr_data = gt[i,:,:,:] # [4 dims because batch, width, height, depth]
        tr_data = np.moveaxis(tr_data, -1, 0)
        tr_list = []
        for sl in tr_data:
            tr_list.append(Image.fromarray(sl,mode='F'))
        tr_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_truth.tif',
                        save_all=True,
                        append_images=tr_list[1:])
        
        all_pred_data = probs[i,:,:] # [4 dims because batch, width, height, depth]
        all_pred_data = np.moveaxis(all_pred_data, -1, 0)
        all_pred_list = []
        for sl in all_pred_data:
            all_pred_list.append(Image.fromarray(sl,mode='F'))
        all_pred_list[0].save(save_dir + '/' + str(step) + '_' + str(i) + '_all_pred.tif',
                        save_all=True,
                        append_images=all_pred_list[1:])
            
def imgs_input_fn(filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(serialized):
        
        keys_to_features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'ct': tf.FixedLenFeature([], tf.string),
            'pt': tf.FixedLenFeature([], tf.string),
            'ctlb': tf.FixedLenFeature([], tf.string),
            'ptlb': tf.FixedLenFeature([], tf.string),
            'bglb': tf.FixedLenFeature([], tf.string)
        }
        features = tf.parse_single_example(serialized, keys_to_features)
        
        # Perform additional preprocessing on the parsed data.
        ct = tf.decode_raw(features['ct'], tf.float32)
        pt = tf.decode_raw(features['pt'], tf.float32)
        ctlb = tf.decode_raw(features['ctlb'], tf.float32)
        ptlb = tf.decode_raw(features['ptlb'], tf.float32)
        bglb = tf.decode_raw(features['bglb'], tf.float32)
        
        #tf.Print(height,['h:', height])
        #tf.Print(width,['w:', width])
        #tf.Print(tf.size(ct),['ct:', tf.size(ct)])
        #tf.Print(tf.size(pt),['pt:', tf.size(pt)])
        #tf.Print(tf.size(ctlb),['ctlb:', tf.size(ctlb)])
        #tf.Print(tf.size(ptlb),['ptlb:', tf.size(ptlb)])
        
        ct = tf.reshape(ct, tf.stack([256, 256, 1]))
        pt = tf.reshape(pt, tf.stack([256, 256, 1]))
        ctlb = tf.reshape(ctlb, tf.stack([256, 256, 2]))
        ptlb = tf.reshape(ptlb, tf.stack([256, 256, 1]))
        bglb = tf.reshape(bglb, tf.stack([256, 256, 1]))
        
        image = tf.concat([ct,pt],2)
        label = tf.concat([bglb,ctlb,ptlb],2)
        
        
        #label = keras.utils.to_categorical(label, num_classes=4)
        #label = tf.one_hot(label,4)
        return ct, pt, label
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size)  # Batch size to use
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times    
    iterator = dataset.make_one_shot_iterator()
    batch_ct, batch_pt, batch_labels = iterator.get_next()
    return batch_ct, batch_pt, batch_labels


def get_scaled_cross_entropy_loss(logits, labels, num_classes, modality_str):
    # Flatten the predictions, so that we can compute cross-entropy for
    # each pixel and get a sum of cross-entropies.
    flat_logits = tf.reshape(tensor=logits, shape=(-1, num_classes + 1))
    flat_label = tf.reshape(tensor=labels, shape=(-1, num_classes + 1))
        
    num_true = tf.count_nonzero(labels, axis=[1,2], keep_dims=True) # how many pixels of each class [1,2 are width/height]
    total_true = tf.reduce_sum(num_true, axis=[1,2,3], keep_dims=True) # total true, calculate per batch, so wipe out all other axes
    scaling = 1 - num_true/total_true
    scaling = tf.cast(scaling, tf.float32)
        
    weighted_labels = tf.multiply(labels, scaling)
    flat_weights = tf.reshape(tensor=weighted_labels, shape=(-1, num_classes + 1))
    weight_map = tf.reduce_sum(flat_weights, axis=1)
        
    # avoid softmax that are 0 using log with an epsilon 
    softmax_val = tf.nn.softmax(flat_logits)
    soft_log = tf.log(softmax_val + 1e-9)
    unweighted_losses = -tf.reduce_sum(flat_label * soft_log, axis=1) 
    weighted_losses = tf.multiply(unweighted_losses, weight_map)
    cross_entropy_str='xentropy_mean_' + modality_str
    cross_entropy_mean = tf.reduce_mean(weighted_losses,name=cross_entropy_str)
    tf.add_to_collection('losses', cross_entropy_mean)
    tf.summary.scalar(cross_entropy_str, cross_entropy_mean)
                
    probabilities = tf.nn.softmax(logits, name=modality_str+'_probabilities') # not flattened!
    prediction = tf.argmax(probabilities, axis=3, name=modality_str+'_prediction') # which axis has the maximum value
    return cross_entropy_mean, probabilities, prediction


mode = sys.argv[1]
data = sys.argv[2]
path_tfrecords_train = _get_tfrecord_files_from_dir(data,'')
if mode=='train':
    shuffle = True
    repeat = 300
else:
    shuffle = False
    repeat = 1
    
input_ct, input_pt, label_gt = imgs_input_fn(path_tfrecords_train, perform_shuffle=shuffle, repeat_count=repeat, batch_size=5)

# patch size
patch_size = 256

# this is the size of our encoded representations
encoding_dim = 256  

kernel = [3,3]
pool = [2,2]

# this is our input placeholder
#input_ct = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,1],name='input_ct')
#input_pt = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,1],name='input_pt')
#label_gt = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,4],name='label_gt')

# encode CT
ct_conv1 = tf.layers.conv2d(
                inputs=input_ct,
                filters=64,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_ct1'
)
ct_conv1_pool = tf.layers.max_pooling2d(
                inputs=ct_conv1,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_ct1'
)
ct_conv2 = tf.layers.conv2d(
                inputs=ct_conv1_pool,
                filters=32,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_ct2'
)
ct_conv2_pool = tf.layers.max_pooling2d(
                inputs=ct_conv2,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_ct2'
)
ct_conv3 = tf.layers.conv2d(
                inputs=ct_conv2_pool,
                filters=16,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_ct3'
)
ct_conv3_pool = tf.layers.max_pooling2d(
                inputs=ct_conv3,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_ct3'
)
ct_conv4 = tf.layers.conv2d(
                inputs=ct_conv3_pool,
                filters=8,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_ct4'
)
ct_conv4_pool = tf.layers.max_pooling2d(
                inputs=ct_conv4,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_ct4'
)
#e_ct = Conv2D(64, (3, 3), activation='relu', padding='same',name='hidden_ct1')(input_ct)
#e_ct = MaxPooling2D((2, 2), padding='same',name='pool_ct1')(e_ct)
#e_ct = Conv2D(32, (3, 3), activation='relu', padding='same',name='hidden_ct2')(e_ct)
#e_ct = MaxPooling2D((2, 2), padding='same',name='pool_ct2')(e_ct)
#e_ct = Conv2D(16, (3, 3), activation='relu', padding='same',name='hidden_ct3')(e_ct)
#e_ct = MaxPooling2D((2, 2), padding='same',name='pool_ct3')(e_ct)
#e_ct = Conv2D(8, (3, 3), activation='relu', padding='same',name='hidden_ct4')(e_ct)
#e_ct = MaxPooling2D((2, 2), padding='same',name='pool_ct4')(e_ct)


# encode PT
pt_conv1 = tf.layers.conv2d(
                inputs=input_pt,
                filters=64,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_pt1'
)
pt_conv1_pool = tf.layers.max_pooling2d(
                inputs=pt_conv1,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_pt1'
)
pt_conv2 = tf.layers.conv2d(
                inputs=pt_conv1_pool,
                filters=32,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_pt2'
)
pt_conv2_pool = tf.layers.max_pooling2d(
                inputs=pt_conv2,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_pt2'
)
pt_conv3 = tf.layers.conv2d(
                inputs=pt_conv2_pool,
                filters=16,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_pt3'
)
pt_conv3_pool = tf.layers.max_pooling2d(
                inputs=pt_conv3,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_pt3'
)
pt_conv4 = tf.layers.conv2d(
                inputs=pt_conv3_pool,
                filters=8,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='hidden_pt4'
)
pt_conv4_pool = tf.layers.max_pooling2d(
                inputs=pt_conv4,
                pool_size=pool,
                strides=pool,
                padding="same",
                name='pool_pt4'
)
#e_pt = Conv2D(64, (3, 3), activation='relu', padding='same',name='hidden_pt1')(input_pt)
#e_pt = MaxPooling2D((2, 2), padding='same',name='pool_pt1')(e_pt)
#e_pt = Conv2D(32, (3, 3), activation='relu', padding='same',name='hidden_pt2')(e_pt)
#e_pt = MaxPooling2D((2, 2), padding='same',name='pool_pt2')(e_pt)
#e_pt = Conv2D(16, (3, 3), activation='relu', padding='same',name='hidden_pt3')(e_pt)
#e_pt = MaxPooling2D((2, 2), padding='same',name='pool_pt3')(e_pt)
#e_pt = Conv2D(8, (3, 3), activation='relu', padding='same',name='hidden_pt4')(e_pt)
#e_pt = MaxPooling2D((2, 2), padding='same',name='pool_pt4')(e_pt)

# average
stacked = tf.stack([ct_conv4_pool,ct_conv4_pool],axis=4,name='concat')
average = tf.reduce_mean(stacked,axis=4,name='fused')
#average = Average(name='average')([e_ct,e_pt])



# reconstruct both
decode1 = tf.layers.conv2d(
                inputs=average,
                filters=8,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='decode_1'
)
width = decode1.get_shape().as_list()[1]
height = decode1.get_shape().as_list()[2]
new_size = [2*width, 2*height]
deconv1 = tf.image.resize_images(
                images=decode1,
                size=new_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
decode2 = tf.layers.conv2d(
                inputs=deconv1,
                filters=16,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='decode_2'
)
width = decode2.get_shape().as_list()[1]
height = decode2.get_shape().as_list()[2]
new_size = [2*width, 2*height]
deconv2 = tf.image.resize_images(
                images=decode2,
                size=new_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
decode3 = tf.layers.conv2d(
                inputs=deconv2,
                filters=32,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='decode_3'
)
width = decode3.get_shape().as_list()[1]
height = decode3.get_shape().as_list()[2]
new_size = [2*width, 2*height]
deconv3 = tf.image.resize_images(
                images=decode3,
                size=new_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
decode4 = tf.layers.conv2d(
                inputs=deconv3,
                filters=64,
                kernel_size=kernel,
                padding="same",
                activation=tf.nn.relu,
                name='decode_4'
)
width = decode4.get_shape().as_list()[1]
height = decode4.get_shape().as_list()[2]
new_size = [2*width, 2*height]
deconv4 = tf.image.resize_images(
                images=decode4,
                size=new_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
reconstructed = tf.layers.conv2d(
                inputs=deconv4,
                filters=4,
                kernel_size=[1,1],
                padding="same",
                activation=None,
                name='reconstructed'
)
#decode = Conv2D(8, (3, 3), activation='relu', padding='same',name='decode_1')(average)
#decode = UpSampling2D((2, 2),name='up_1')(decode)
#decode = Conv2D(16, (3, 3), activation='relu', padding='same',name='decode_2')(decode)
#decode = UpSampling2D((2, 2),name='up_2')(decode)
#decode = Conv2D(32, (3, 3), activation='relu', padding='same',name='decode_3')(decode)
#decode = UpSampling2D((2, 2),name='up_3')(decode)
#decode = Conv2D(64, (3, 3), activation='relu', padding='same',name='decode_4')(decode)
#decode = UpSampling2D((2, 2),name='up_4')(decode)
#reconstruced = Conv2D(4, (3, 3), activation='linear', padding='same',name='reconstructed')(decode)

all_cost, all_probabilities, all_pred = get_scaled_cross_entropy_loss(reconstructed,
    label_gt, 3, 'all')  

optimizer = tf.train.GradientDescentOptimizer(0.001)
train_op = optimizer.minimize(all_cost)
eval_op = all_probabilities

g_init_op = tf.global_variables_initializer()
l_init_op = tf.local_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count = {'GPU': 1})) as mon_sess: 
    # Need a saver to save and restore all the variables.
    saver = tf.train.Saver()
    mon_sess.run([g_init_op, l_init_op])
    
    if mode == 'train':
        tstep = 1
        while True:
            try:
                mon_sess.run(train_op)
                tstep = tstep + 1
                if tstep % 100 == 0:
                    print('ON STEP %d' % tstep)
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                save_path = saver.save(mon_sess, 'baseline_vantulder_convae/run-end.ckpt')
                print('Model saved in path: %s' % save_path)
                break
    else:
        step = 1
        while True:
            ckpt_meta = 'baseline_vantulder_convae/run-end.ckpt.meta'
            meta_restore = tf.train.import_meta_graph(ckpt_meta)
    
            try:
                ckpt_state = tf.train.get_checkpoint_state('baseline_vantulder_convae')
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
                sys.exit(0)
            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model to eval yet at %s', 'baseline_vantulder_convae')
                sys.exit(0)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            #ckpt_saver.restore(mon_sess, ckpt_state.model_checkpoint_path) 
            meta_restore.restore(mon_sess, ckpt_state.model_checkpoint_path) 
                           
            try:    
                num_images = 5
                probs, gt = mon_sess.run([eval_op, label_gt])
                _saveImages(num_images, step, gt, probs)
                step = step + 1
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                break



'''
# create the model
model = Model(inputs=[input_ct, input_pt], outputs=[reconstruced_ct, reconstruced_pt])

# define cost
sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss=['mean_squared_error','mean_squared_error'],
              metrics=['accuracy', root_mean_squared_error]) # needs rsme fix

model.summary()


estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
path_tfrecords_train = _get_tfrecord_files_from_dir('../data_trn','')
print('Training with %d studies' % len(path_tfrecords_train))
#path_tfrecords_train = ['../patch_train_data/M00570.tfrecords']
path_tfrecords_test_all = _get_tfrecord_files_from_dir('../data_tst','')
#path_tfrecords_test_pos = _get_tfrecord_files_from_dir('../patch_test_data','pos')
#path_tfrecords_test_lung = _get_tfrecord_files_from_dir('../patch_test_data','lung')
#path_tfrecords_test_med = _get_tfrecord_files_from_dir('../patch_test_data','med')
#path_tfrecords_test_tumour = _get_tfrecord_files_from_dir('../patch_test_data','tumour')
#path_tfrecords_test = ['../patch_train_data/M00093.tfrecords']
train_batch_size=50
train_epochs=300


print('TRAINING')
estimator.train(input_fn=lambda: imgs_input_fn(path_tfrecords_train, perform_shuffle=True, repeat_count=train_epochs,batch_size=train_batch_size))

# try to export #
#exported_model = estimator.export_savedmodel(export_dir_base = './export', serving_input_receiver_fn = serving_input_fn)


#print('TESTING ALL')
#score_all = estimator.evaluate(input_fn=lambda: imgs_input_fn(path_tfrecords_test_all, perform_shuffle=False, repeat_count=1,batch_size=1))
#print(score_all)


print('EXPORTING')
# try to export #
exported_model = estimator.export_savedmodel(export_dir_base = './export_autoeconder', serving_input_receiver_fn = serving_input_fn)
print('EXPORTED')


# for debug - train USING RANDOM DATA!
#data_ct = np.random.random( (5,patch_size,patch_size,1) )
#data_pt = np.random.random( (5,patch_size,patch_size,1) )
#data = [data_ct, data_pt]
#labels = data

# Train the model, iterating on the data in batches of 1 samples
#model.fit(data, labels, epochs=5, batch_size=1)

#model.save('autoencoder.h5')
'''