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
    true_positives = tf.reduce_sum(y_true * y_pred)
    #true_positives = K.sum(y_true == y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    #predicted_positives = K.sum(y_pred)
    precision = true_positives / (predicted_positives + 1e-9)
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
    save_dir = 'baseline_zhang_seg/outputs'
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

def serving_input_fn():
    feature_spec = {'conv2d_1_input': tf.FixedLenFeature(dtype=tf.float32, shape=[256,256,2])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()




    
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
        return image, label
    
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
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

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
    train = True
else:
    shuffle = False
    repeat = 1
    train = False
    
    
train_flag = tf.placeholder(tf.bool,name='train_flag')    
input_img, label_gt = imgs_input_fn(path_tfrecords_train, perform_shuffle=shuffle, repeat_count=repeat, batch_size=5)

conv1 = tf.layers.conv2d(
                inputs=input_img,
                filters=64,
                kernel_size=[5,5],
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.ones_initializer(),
                padding="valid",
                activation=tf.nn.relu,
                name='conv1'
)
drop1 = tf.layers.dropout(
    inputs=conv1,
    rate=0.5,
    training=train_flag
)
conv2 = tf.layers.conv2d(
                inputs=drop1,
                filters=256,
                kernel_size=[5,5],
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.ones_initializer(),
                padding="valid",
                activation=tf.nn.relu,
                name='conv2'
)
drop2 = tf.layers.dropout(
    inputs=conv2,
    rate=0.5,
    training=train_flag
)
conv3 = tf.layers.conv2d(
                inputs=drop2,
                filters=768,
                kernel_size=[5,5],
                kernel_initializer=tf.random_normal_initializer(),
                bias_initializer=tf.ones_initializer(),
                padding="valid",
                activation=tf.nn.relu,
                name='conv3'
)
drop3 = tf.layers.dropout(
    inputs=conv3,
    rate=0.5,
    training=train_flag
)

lrn = tf.nn.lrn(
    drop3,
    alpha=1e-4, 
    bias=2,
    beta=0.75,
    depth_radius=5,
    name='lrn'
)

deconv1 = tf.layers.conv2d_transpose(
    inputs=lrn,
    filters=256,
    kernel_size=[5,5],
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.random_normal_initializer,
    bias_initializer=tf.ones_initializer(),
    name='deconv1'
)
drop4 = tf.layers.dropout(
    inputs=deconv1,
    rate=0.5,
    training=train_flag
)
deconv2 = tf.layers.conv2d_transpose(
    inputs=drop4,
    filters=64,
    kernel_size=[5,5],
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.random_normal_initializer,
    bias_initializer=tf.ones_initializer(),
    name='deconv2'
)
drop5 = tf.layers.dropout(
    inputs=deconv2,
    rate=0.5,
    training=train_flag
)
deconv3 = tf.layers.conv2d_transpose(
    inputs=drop5,
    filters=4,
    kernel_size=[5,5],
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=tf.random_normal_initializer,
    bias_initializer=tf.ones_initializer(),
    name='deconv3'
)
drop6 = tf.layers.dropout(
    inputs=deconv3,
    rate=0.5,
    training=train_flag
)

all_cost, all_probabilities, all_pred = get_scaled_cross_entropy_loss(drop6,
    label_gt, 3, 'all') 

prec = precision(label_gt,tf.one_hot(all_pred,4))

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
                _, p = mon_sess.run([train_op, prec], feed_dict={train_flag: train})
                tstep = tstep + 1
                if tstep % 100 == 0:
                    print('ON STEP %d PRECISION IS %d' % (tstep, p))
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                save_path = saver.save(mon_sess, 'baseline_zhang_seg/run-end.ckpt')
                print('Model saved in path: %s' % save_path)
                break
    else:
        step = 1
        while True:
            ckpt_meta = 'baseline_zhang_seg/run-end.ckpt.meta'
            meta_restore = tf.train.import_meta_graph(ckpt_meta)
    
            try:
                ckpt_state = tf.train.get_checkpoint_state('baseline_zhang_seg')
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
                sys.exit(0)
            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model to eval yet at %s', 'baseline_zhang_seg')
                sys.exit(0)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            #ckpt_saver.restore(mon_sess, ckpt_state.model_checkpoint_path) 
            meta_restore.restore(mon_sess, ckpt_state.model_checkpoint_path) 
                           
            try:    
                num_images = 5
                probs, gt = mon_sess.run([eval_op, label_gt], feed_dict={train_flag: train})
                _saveImages(num_images, step, gt, probs)
                step = step + 1
            except tf.errors.OutOfRangeError:
                print('OUT OF DATA - ENDING')
                break


'''
# starts here
model = Sequential()

# define input - patches of 13x13x2 in unknown number of batches
input_shape=(256,256,2,)

# define model
model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Dropout(0.5))
#model.add(Conv2D(768, (1, 1), activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Conv2D(768, (5, 5), activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Dropout(0.5))
model.add(LRN2D())
model.add(Conv2DTranspose(256, (5,5), activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Dropout(0.5))
model.add(Conv2DTranspose(64, (5,5), activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Dropout(0.5))
model.add(Conv2DTranspose(4, (5,5), activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
model.add(Dropout(0.5))
model.add(Activation('softmax',name='prob'))
#model.add(Dense(4, activation='relu', kernel_initializer='random_normal', bias_initializer='ones'))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Activation('softmax',name='prob'))

# define cost
sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy', precision, recall, root_mean_squared_error]) # needs rsme fix

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


print('TESTING ALL')
score_all = estimator.evaluate(input_fn=lambda: imgs_input_fn(path_tfrecords_test_all, perform_shuffle=False, repeat_count=1,batch_size=1))
print(score_all)

#print('TESTING POS')
#score_pos = estimator.evaluate(input_fn=lambda: imgs_input_fn(path_tfrecords_test_pos, perform_shuffle=False, repeat_count=1,batch_size=1))
#print(score_pos)

#print('TESTING LUNG')
#score_lung = estimator.evaluate(input_fn=lambda: imgs_input_fn(path_tfrecords_test_lung, perform_shuffle=False, repeat_count=1,batch_size=1))
#print(score_lung)

#print('TESTING MEDIASTINUM')
#score_med = estimator.evaluate(input_fn=lambda: imgs_input_fn(path_tfrecords_test_med, perform_shuffle=False, repeat_count=1,batch_size=1))
#print(score_med)

#print('TESTING TUMOUR')
#score_tum = estimator.evaluate(input_fn=lambda: imgs_input_fn(path_tfrecords_test_tumour, perform_shuffle=False, repeat_count=1,batch_size=1))
#print(score_tum)

print('EXPORTING')
# try to export #
exported_model = estimator.export_savedmodel(export_dir_base = './export_seg', serving_input_receiver_fn = serving_input_fn)
print('EXPORTED')


### THE FOLLOWING IS PROBABLY NOT NEEDED
#train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(path_tfrecords_train,
#                                                                   perform_shuffle=True,
#                                                                   repeat_count=1,
#                                                                   batch_size=128), 
#                                    max_steps=500)
#eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(path_tfrecords_test,
#                                                                 perform_shuffle=False,
#                                                                 batch_size=1))

#tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
### THE ABOVE IS PROBABLY NOT NEEDED

# for debug - train USING RANDOM DATA!
#data = np.random.random( (100,13,13,2) )
#labels = np.random.randint(4, size=(100, 1)) # use centre pixel as label

# Convert labels to categorical one-hot encoding
#one_hot_labels = keras.utils.to_categorical(labels, num_classes=4)

# Train the model, iterating on the data in batches of 32 samples
#model.fit(data, one_hot_labels, epochs=1, batch_size=5)
'''




