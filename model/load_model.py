import tensorflow as tf


def load_model(path='model/model_savings'):
    return tf.keras.models.load_model(path)