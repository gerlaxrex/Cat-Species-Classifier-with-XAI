import tensorflow as tf
import matplotlib.pyplot as plt
from configs import IMG_SHAPE
import numpy as np
import matplotlib.cm as cm

def make_gradcam_heatmap(img, model):
    """Generate the heatmap for using the GradCAM algorithm"""
    with tf.GradientTape() as tape:
        preds, last_conv_layer_output = model(img, training=False)
        pred_index = tf.argmax(tf.nn.softmax(preds[0]))
        class_channel = tf.convert_to_tensor(preds[:, pred_index])

    grads = tape.gradient(class_channel, last_conv_layer_output)
    #GPA
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def transform_heatmap(heatmap):
    """Transform the heatmap obtained from the gradients into a JET image for visualization purposes"""
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(IMG_SHAPE)
    return jet_heatmap

def visualize_result(img, heatmap):
    fig, ax = plt.subplots(1,3,figsize=(20,10))
    jet_heatmap = transform_heatmap(heatmap)
    ax[0].imshow(img)
    ax[1].imshow(jet_heatmap)
    ax[2].imshow(img)
    ax[2].imshow(jet_heatmap, alpha = 0.3)
    plt.imshow()
