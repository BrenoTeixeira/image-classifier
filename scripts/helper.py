import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn, from_numpy, utils
from torchvision import transforms, datasets


def preprocess(image_path):
    """This function receives the path for image and transforms it making it ready for model prediction.

    Args:
        image_path (str): Path to the image.

    Returns:
        image (tensor): The tensor of the transformed image.
    """

    image = Image.open(image_path)
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image - mean) / std
    image = from_numpy(image.transpose(2, 0, 1))

    return image


def imshow(image, ax=None, title=None):
    """Image for Tensor.

    Args:
        image (tensor): Transformed imager tensor.
        ax (Axes, optional): Axis. Defaults to None.
        title (str, optional): Plot's Title. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)
    return ax


def load_final_layer(hidden_layers):
    """This function uses the hidden layers of the FeedForward of the trained model to rebuild its structure and use it for prediction.

    Args:
        hidden_layers (list): List with the structure of the saved FeedForward Classifier.

    Returns:
        nn.Sequential: Sequential structure for the FeedForward Classifier.
    """

    classifier = nn.Sequential(*hidden_layers)

    return classifier
