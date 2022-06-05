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

        self.conv2d_16_3x3_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')
        self.conv2d_32_3x3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.conv2d_32_3x3_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.conv2d_64_3x3_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv2d_64_3x3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv2d_64_5x5_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')
        self.maxpool2d_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.resnet = tf.keras.applications.VGG19(weights='imagenet', include_top=False)

        self.resnet.trainable = False

        self.gap_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_layer = tf.keras.layers.Dense(units=256, activation='relu')
        self.prediction_layer = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        # Preprocessing
        x = self.preprocessing_layer(inputs, training=training)
        # Convolutional Network

        # x = self.conv2d_16_3x3_1(x) # 128x128x3 -> 126x126x16
        # x = self.conv2d_32_3x3_1(x) # 126x126x16 -> 124x124x32
        # x = self.maxpool2d_layer(x) # 124x124x32 -> 62x62x32
        # x = self.conv2d_32_3x3_2(x) # 62x62x32 -> 60x60x32
        # x = self.conv2d_64_3x3_1(x) # 60x60x32 -> 58x58x64
        # x = self.maxpool2d_layer(x) # 58x58x64 -> 29x29x64
        # x = self.conv2d_64_3x3_2(x) # 29x29x64 -> 27x27x64
        # x = self.conv2d_64_5x5_1(x) # 27x27x64 -> 23x23x64
        # x = self.maxpool2d_layer(x) # 23x23x64 -> 11x11x64
        x = self.resnet(x, training=training)
        # Prediction Network
        x = self.gap_layer(x)  # 11x11x64 -> 1x1x64
        x = self.dense_layer(x)
        return self.prediction_layer(x)