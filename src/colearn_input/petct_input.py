import tensorflow as tf

def build_input(data_paths, batch_size, num_epochs, im_size, train_mode):
    """Build PET-CT image and labels.

    Args:
        data_paths: Filenames for data.
        batch_size: Input batch size.
        train_mode: boolean, True if training.
    Returns:
        ct, pt: Batches of images. 
        ctlb, ptlb: Batches of labels. 
    """
    
    dataset = tf.data.TFRecordDataset(data_paths)
    
    # Use `tf.parse_single_example()` to extract data from a `tf.Example` protocol buffer, and perform per-record preprocessing.
    def parser(record):
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
        features = tf.parse_single_example(record, keys_to_features)
        
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        # Perform additional preprocessing on the parsed data.
        ct = tf.decode_raw(features['ct'], tf.float32)
        pt = tf.decode_raw(features['pt'], tf.float32)
        ctlb = tf.decode_raw(features['ctlb'], tf.float32)
        ptlb = tf.decode_raw(features['ptlb'], tf.float32)
        bglb = tf.decode_raw(features['bglb'], tf.float32)
        
        
        ct = tf.reshape(ct, tf.stack([height, width, 1]))
        pt = tf.reshape(pt, tf.stack([height, width, 1]))
        ctlb = tf.reshape(ctlb, tf.stack([height, width, 2]))
        ptlb = tf.reshape(ptlb, tf.stack([height, width, 1]))
        bglb = tf.reshape(bglb, tf.stack([height, width, 1]))                
        
        # send the data for augmentation
        modalities, labels = augment(ct, pt, ctlb, ptlb, bglb, im_size, train_mode)
        return modalities, labels
        

    dataset = dataset.map(parser)
    if train_mode:
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=500)
    return dataset, ['CT', 'PET']
    
    

    

def augment(ct, pt, ctlb, ptlb, bglb, image_size, train_mode):
    with tf.variable_scope('image_data'):
        image = tf.concat([ct, pt, ctlb, ptlb, bglb],2) # concat along 3rd dimension [height, width, depth]
        image_depth = image.get_shape()[2]     
    
        if train_mode: 
            # data augmentation - crop and flip
            paddings = tf.constant([[10,10], [10,10], [0,0]])
            image = tf.pad(image, paddings, "REFLECT")                
            image = tf.random_crop(image, [image_size, image_size, image_depth])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        
            ct = tf.slice(image, [0, 0, 0], [image_size, image_size, 1], name='ct_slice')
            pt = tf.slice(image, [0, 0, 1], [image_size, image_size, 1], name='pt_slice')
            lb = tf.slice(image, [0, 0, 2], [image_size, image_size, 4], name='lb_slice')
        else:
            ct = tf.slice(image, [0, 0, 0], [image_size, image_size, 1], name='ct_slice')
            pt = tf.slice(image, [0, 0, 1], [image_size, image_size, 1], name='pt_slice')
            lb = tf.slice(image, [0, 0, 2], [image_size, image_size, 4], name='lb_slice')
            
            
        # now standardize only PET and CT
        ct = tf.image.per_image_standardization(ct)
        pt = tf.image.per_image_standardization(pt)    
        return (ct, pt), lb
