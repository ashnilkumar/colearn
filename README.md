# colearn
A deep learning technique for the spatially varying fusion of spatially-aligned feature maps from multi-modality medical images.

## Dependencies
- tensorflow-gpu 1.10+
- CUDA 9.0
- CuDNN 7.0
- python3
- numpy
- Pillow

This version has only been tested on Ubuntu 16.04 on an NVIDIA GTX 1080 Ti.

## Usage
An example architecture is included [here](src/colearn_cnn.py). A main function to use the architecture on PET-CT data is also [included](src/colearn_cnn_petct_main.py).

### Data Preparation
The example main file expects PET-CT lung data in the TFRecord format.

```python
	'height': tf.FixedLenFeature([], tf.int64)	# height of the image
        'width': tf.FixedLenFeature([], tf.int64)	# width of the image
        'depth': tf.FixedLenFeature([], tf.int64)	# deprecated will be removed
        'ct': tf.FixedLenFeature([], tf.string)		# CT image (height * width * 1)
        'pt': tf.FixedLenFeature([], tf.string)		# PET image (height * width * 1)
        'ctlb': tf.FixedLenFeature([], tf.string)	# binary masks of anatomical regions (height * width * 2)
        'ptlb': tf.FixedLenFeature([], tf.string)	# binary mask of tumours (height * w * 1)
        'bglb': tf.FixedLenFeature([], tf.string)	# binary mask of background (height * width * 1)
```

For the anatomical binary masks, the two channels (depth dimension of 2) are for the lung fields and the mediastinum.

### Training
To train using default parameters on a GPU, use:

```
python3 colearn_cnn_petct_main.py \
	--mode=train \
	--train_data_path=TRAIN_DATA_PATH \
	[--valid_data_path=VALID_DATA_PATH] \
	--log_root=LOG_ROOT \
	--train_dir=LOG_ROOT/TRAIN_DIR \
	[--valid_dir=LOG_ROOT/VALID_DIR]	
```

where:

- `TRAIN_DATA_PATH` is the directory with the \*.tfrecord files for the training data
- `VALID_DATA_PATH` is the directory with the \*.tfrecord files for the validation data
- `LOG_ROOT` is the directory where the logs for this run will be stored
- `TRAIN_DIR` is the subdirectory of `LOG_ROOT` for training logs
- `VALID_DIR` is the subdirectory of `LOG_ROOT` for validation logs

If `VALID_DATA_PATH` is not specified, there will be no outputs for model validation.
If `VALID_DATA_PATH` is specified, then `VALID_DIR` must also be specified.

### Evaluation
To evaluate your trained model, use:

```
python3 colearn_cnn_petct_main.py \
	--mode=eval \
	--eval_data_path=EVAL_DATA_PATH \
	--log_root=LOG_ROOT \
	--eval_dir=LOG_ROOT/EVAL_DIR \
	--checkpoint_to_eval=CKPT_TO_EVAL \
```

where:

- `EVAL_DATA_PATH` is the directory with the \*.tfrecord files for the evaluation data
- `LOG_ROOT` is the directory where the logs for this run will be stored
- `EVAL_DIR` is the subdirectory of `LOG_ROOT` for evaluation logs
- `CKPT_TO_EVAL` is the integer number of the training checkpoint to evaluate  
    
## Citation
If you make use of this code in your work, please cite the following paper:
    `A. Kumar, M. Fulham, D. Feng, and J. Kim, "Co-Learning Feature Fusion Maps from PET-CT Images of Lung Cancer", _JOURNAL_ **VOL**(NO):pages, year.`
