import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from configs import *

# Read and process images for creating the datasets

preprocessed_images = []
filenames = []
labels = []

logger = logging.getLogger('DatasetFactory')


def preprocess_dataset(directory: str):
    logger.info(f'Starting process from {directory}')
    for (root, dirs, files) in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file).replace('\\', '/')
            filenames.append(img_path)
            preprocessed_images.append(decode_image(img_path))
            labels.append(retrieve_label(img_path))


def decode_image(img_path):
    with Image.open(img_path) as img:
        p_img = np.array(img.resize(IMG_SHAPE).convert('RGB'))
        return p_img


def retrieve_label(img_path):
    return CLASS_DICT[img_path.split('/')[-2]]


# Function for dataset mapping if the dataset is composed of (filename, label)
def process_image(img_path):
    """Decode, resize and convert to RGB image"""
    with Image.open(img_path.numpy()) as img:
        p_img = np.array(img.resize(IMG_SHAPE).convert('RGB'))
        return tf.convert_to_tensor(p_img)


def process_label(img_path):
    """Retrieve the label for the image"""
    parts = tf.strings.split(img_path, '/')
    one_hot = parts[-2] == CLASS_LIST
    return tf.argmax(one_hot)


def build_dataset(images, labels=None, batch_size=100, shuffle=True, prefetch_size=1, seed=None):
    """Build a dataset with the labels (if present), shuffling, batching and prefetching"""
    # Build the single datasets
    if labels is None:
        logger.info('Start creation of dataset (unsupervised)...')
        tot_ds = tf.data.Dataset.from_tensor_slices(images)
    else:
        logger.info('Start creation of dataset (supervised)...')
        tot_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # tot_ds = tot_ds.map(lambda x,l : (tf.py_function(process_image, [x], tf.uint8), l))
    # Transform into one and split in train/validation
    total_size = tot_ds.cardinality().numpy()
    # Shuffle, batch, optimize
    tot_ds = tot_ds.repeat()
    if shuffle:
        tot_ds = tot_ds.shuffle(min(total_size, 1000), seed=seed)
    tot_ds = tot_ds.batch(batch_size).prefetch(prefetch_size)
    logger.info(f'Created dataset with {total_size} samples.')
    return tot_ds, total_size


def split_train_valid_datasets(ds, labels, train_size=0.8, seed=None):
    """Function for creating a train_test split with tensorflow.
    The function performs three main steps:
        - Creates an indexed array to be splitted with the sklearn.preprocessing function 'train_test_split'
          with the stratification based on the labels.
        - Then uses the '.enumerate()' functionality over a tensorflow Dataset
        - Then uses a '.filter()' function in order to select ONLY the indexes contained in the train OR validation split
        - Then uses a map in order to remove the index created by the '.enumerate()' function.
    """
    X_indexes = np.array(list(range(len(labels))))
    X_train, y_train, X_valid, y_valid = train_test_split(X_indexes,
                                                          labels,
                                                          stratify=labels,
                                                          shuffle=True,
                                                          train_size=train_size,
                                                          random_state=seed)

    def select_index_train(el):
        return tf.math.reduce_any(el[0] == X_train)

    def select_index_valid(el):
        return tf.math.reduce_any(el[0] == X_train)

    def remove_index(el):
        return el[1]
    logger.info(f'Generating training dataset...')
    train_ds = ds.enumerate().filter(select_index_train).map(remove_index)
    logger.info(f'Generated training dataset!')

    logger.info(f'Generating validation dataset...')
    valid_ds = ds.enumerate().filter(select_index_valid).map(remove_index)
    logger.info(f'Generated validation dataset!')

    return (train_ds, len(X_train)), (valid_ds, len(X_valid))
