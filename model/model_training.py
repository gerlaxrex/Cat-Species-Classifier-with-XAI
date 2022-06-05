from classifier import CatSpeciesClassifier
from configs import *
import tensorflow as tf
from typing import Tuple


def prepare_model() -> tf.keras.Model:
    # Define the model
    cat_species_classifier = CatSpeciesClassifier(input_shape=IMG_SHAPE, num_classes=len(CLASS_DICT.keys()))
    # Define the loss function, optimizer and metrics
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    metrics_array = ['accuracy']
    # Build and compile the model
    cat_species_classifier.build(input_shape=(None, *IMG_SHAPE, 3))
    cat_species_classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                   optimizer='adam',
                                   metrics=metrics_array)

    return cat_species_classifier


def train_model(model: tf.keras.Model,
                train_ds: tf.data.Dataset,
                valid_ds: tf.data.Dataset,
                spe_train: int,
                spe_valid: int) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    # Define the possible callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
    # Fit the model
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        steps_per_epoch=spe_train,
                        validation_data=valid_ds,
                        validation_steps=spe_valid,
                        callbacks=callbacks)
    return model, history
