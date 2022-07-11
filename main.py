import logging
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from cam.cam_utils import perform_cam
from configs import *
from data.datasets import preprocess_dataset, preprocessed_images, labels, build_dataset
from model.load_model import load_model
from model.model_training import train_model, prepare_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# General configurations
SAVE_MODEL = False
TRAIN_MODEL = True
SAVING_PATH = 'model/model_savings'
MULTIPLIER = 10
SEED = 42

logging.basicConfig(level=logging.INFO,
                    format='(%(name)s) - %(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Prepare the datasets for training and validation
    img_dir = './data/Felidae'
    logger.info('Starting script. Cat Species classifier.')
    logger.info(f'Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))} ({tf.config.list_physical_devices("GPU")})')
    if TRAIN_MODEL:
        preprocess_dataset(img_dir)

        # Train/Valid split
        train_images, valid_images, train_labels, valid_labels = train_test_split(preprocessed_images,
                                                                                  labels,
                                                                                  train_size=0.8,
                                                                                  stratify=labels,
                                                                                  random_state=SEED)

        # Alternative train_validation splitting
        # total_ds = tf.data.Dataset.from_tensor_slices((preprocessed_images, labels))
        # train_ds, train_size, valid_ds, valid_size = split_train_valid_datasets(total_ds,
        #                                                                         labels,
        #                                                                         train_size=0.8,
        #                                                                         seed=SEED)

        # Build the datasets
        train_ds, train_size = build_dataset(train_images, train_labels, TRAIN_BS, seed=SEED)
        valid_ds, valid_size = build_dataset(valid_images, valid_labels, VALID_BS, shuffle=False, seed=SEED)

        # Define the Steps for training and validation
        SPE_TRAIN = train_size*MULTIPLIER // TRAIN_BS
        SPE_VALID = valid_size*MULTIPLIER // VALID_BS

        # Define, build and compile the model
        cat_species_classifier = prepare_model()
        logger.info('Starting training')
        logger.info('Training model.')
        cat_species_classifier, history = train_model(cat_species_classifier, train_ds, valid_ds, SPE_TRAIN, SPE_VALID)
        if SAVE_MODEL:
            logger.info('Saving model.')
            cat_species_classifier.save(SAVING_PATH, save_format='tf')
            logger.info('Model has been correctly saved.')
    else:
        logger.info('Attempting to load model...')
        cat_species_classifier = load_model(SAVING_PATH)
        logger.info('Model loaded')


    # Test and visualize the cam model
    class_type = 'Cheetah'
    number = 50
    test_img = f'./data/Felidae/{class_type}/{class_type}_{number:03d}.jpg'

    logger.info(f'Testing image {test_img}')
    perform_cam(test_img, cat_species_classifier, class_type)

    # Visualize the GradCAM algorithm result












