import tensorflow as tf


class PreprocessingLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__(name='Preprocessing_layer')
        self.rescale_layer = tf.keras.layers.Rescaling(scale=1. / 255)
        self.rotate_layer = tf.keras.layers.RandomRotation(factor=0.2)
        self.zoom_layer = tf.keras.layers.RandomZoom(height_factor=0.2)
        self.contrast_layer = tf.keras.layers.RandomContrast(factor=0.2)

    def call(self, inputs, training=False):
        x = self.rescale_layer(inputs)
        x = self.rotate_layer(inputs, training=training)
        x = self.zoom_layer(x, training=training)
        x = self.contrast_layer(x, training=training)
        return x


class CatSpeciesClassifier(tf.keras.Model):

    def __init__(self, input_shape, num_classes):
        super().__init__(name='CatSpecies_Classifier')

        self.preprocessing_layer = PreprocessingLayer()
        self.vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
        self.vgg19.trainable = False

        # Create the intermediate model
        self.vgg19_double_out = tf.keras.Model(inputs=[self.vgg19.inputs],
                                               outputs=[self.vgg19.get_layer(index=-2).output, self.vgg19.output])

        self.gap_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_layer = tf.keras.layers.Dense(units=256, activation='relu')
        self.prediction_layer = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        # Preprocessing
        x = self.preprocessing_layer(inputs, training=training)
        # Convolutional Network
        last_conv_out, x = self.vgg19_double_out(x, training=training)
        # Prediction Network
        x = self.gap_layer(x)  # 11x11x64 -> 1x1x64
        x = self.dense_layer(x)
        return self.prediction_layer(x), last_conv_out

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred, _ = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
