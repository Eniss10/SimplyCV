import tensorflow as tf
import matplotlib.pyplot as plt

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads an image from filename, turns it into tensor and reshapes into
    (224, 224, 3).

    Parameters:
    - filename (str): string filename of target image.
    - img_shape (int): size to resize target image to, default 224.
    - sclale (bool): whether to scale pixel values to range(0, 1), default True.
    
    Returns:
    - TensorFlow Tensor representing the processed image.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img/255.0
    else:
        return img

def plot_loss_curves(history):
    """
    Returns seperate loss curves for training and validation metrics.

    Args:
    - history: TensorFlow model History object.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();