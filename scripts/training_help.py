from torch import nn, load, save
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch import nn, utils


def train_transforms():
    """This function defines and returns a transformer to preprocess an image train dataset.

    Returns:
        Return a transformer define with transforms.Compose from torchvision
    """

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms


def test_transforms():
    """This function defines and returns a transformer to preprocess an image validation dataset.

    Returns:
        Return a transformer define with transforms.Compose from torchvision
    """

    test_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ]
    )

    return test_transforms


def load_data(data_dir, class_to_index=False):
    """This function receives the path for the datasets and a class_to_index flag. It returns the trainloader, valloader and testloader or class_to_idx form the train_data.

    Args:
        data_dir (str): Directory path where the train, validation and test sets are stored.
        class_to_index (bool): Flag to change what the function returns. If set to true, the function return a dictionary mapping the index to class.

    Returns:
        Datasets (DataLoader): Train, Validation and Test dataloaders.
        or
        class_to_idx (dict): Dictionary that maps the index to a category class name.
    """

    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms())
    val_data = datasets.ImageFolder(valid_dir, transform=test_transforms())
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms())

    trainloader = utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = utils.data.DataLoader(val_data, batch_size=64)
    testloader = utils.data.DataLoader(test_data, batch_size=64)

    if class_to_index:
        return train_data.class_to_idx

    else:
        return trainloader, valloader, testloader


def checkpoint(model, optimizer, filename, epoch, class_to_index, arch):
    """This function saves a checkpoint of the training model (optimezer state, model state and hidden layers).

    Args:
        model (torchvision.model): Trained model (VGG, ResNet or AlexNet).
        optimizer (torch.optim): Optimizer used to train the model (Adam, AdamW, Rprop, etc.).
        filename (string): Path to save the checkpoint (end with name.pth).
        epoch (int): Epoch at which the training stopped.
    """
    if arch == 'resnet':
        hidden_layers = [each for each in model.fc.children()]
    else:
        hidden_layers = [each for each in model.classifier.children()]
        
    checkpoint = {
        "epoch": epoch + 1,
        "arch": arch,
        "hidden_layers": hidden_layers,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_index,
        "optimizer": optimizer.state_dict(),
    }

    save(checkpoint, filename)


def load_checkpoint(filename):
    """Load a checkpoint of a trained model.

    Args:
        filename (string): Path to the saved checkpoint of the trained model.

    Returns:
        dict: Dictionary with the model's saved parameters and other attributes.
    """

    checkpoint = load(filename)

    return checkpoint


def final_layer(hidden_units, number_of_categories):
    """This function defines a structure for the FeedForward Classifier.

    Args:
        hidden_units (int): Hidden units of the first hidden layer.
        number_of_categories (int): Number of units in the final layer output.

    Returns:
        Sequential: Sequential structure for the FeedForward Classifier.
    """

    # Final Layer Classifier
    classifier = nn.Sequential(
        nn.Linear(hidden_units, hidden_units // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units // 2, hidden_units // 4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units // 4, number_of_categories),
        nn.LogSoftmax(dim=1),
    )

    return classifier
