import tensorflow as tf
import numpy as np
from PIL import Image
import os
from configs import *

# Read and process images for creating the datasets

preprocessed_images = []
filenames = []
labels = []


def preprocess_dataset(directory: str):
    for (root, dirs, files) in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file)
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


def build_dataset(images, labels=None, batch_size=100, shuffle=True, prefetch_size=1):
    """Build a dataset with the labels (if present), shuffling, batching and prefetching"""
    # Build the single datasets
    if labels is None:
        tot_ds = tf.data.Dataset.from_tensor_slices(images)
    else:
        tot_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # tot_ds = tot_ds.map(lambda x,l : (tf.py_function(process_image, [x], tf.uint8), l))
    # Transform into one and split in train/validation
    total_size = tot_ds.cardinality().numpy()
    # Shuffle, batch, optimize
    tot_ds = tot_ds.repeat()
    if shuffle:
        tot_ds = tot_ds.shuffle(total_size)
    tot_ds = tot_ds.batch(batch_size).prefetch(prefetch_size)
    return tot_ds, total_size
