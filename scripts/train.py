# Imports here
import click
import torch
import numpy as np
from torch import nn
from torch import optim
from pathlib import Path
from torchvision import models
from helper import load_final_layer
from training_help import checkpoint, load_checkpoint, final_layer, load_data


def train_model(
    trainloader,
    valloader,
    class_to_idx,
    save_dir,
    model,
    arch,
    gpu,
    epochs,
    learning_rate,
    resume_training=False,
):
    """This function trains a Network on an image dataset.

    Args:
        trainloader: Train set of images.
        valloader: Validation set of images.
        save_dir (str): Directory path to save the model's checkpoint.
        model (torchvision.model): Pretrained model with new classifier for tuning (VGG, ResNet or AlexNet).
        gpu (bool): Flag for allowing the use of GPU as the device.
        epochs (int, optional): Number of epochs to train the model. Defaults to 5.
        learning_rate (float, optional): Model's learning rate parameter. Defaults to 0.001.
        resume_training (bool, optional): Flag to continue training from the last epoch. Defaults to False.
    """

    # Defining the device to use.
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    criterion = nn.NLLLoss()

    # optimizer
    if arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if resume_training:
        optimizer.load_state_dict(load_checkpoint(save_dir)["optimizer"])
        start_epoch = load_checkpoint(save_dir)['epoch']
    else:
        start_epoch = 0

    model.to(device)
    
    step = 1
    print_every = 5
    running_loss = 0

    # training iteration
    for epoch in range(start_epoch, epochs):

        for images, labels in trainloader:
            step += 1

            images, labels = images.to(device), labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()

            # Update weights
            optimizer.step()
            running_loss += loss.item()

            # Validadtion Iteration
            if step % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0
                for images, labels in valloader:

                    # setting images to the device
                    images, labels = images.to(device), labels.to(device)

                    logps = model.forward(images)

                    # Getting predicted Labels with topk
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equal = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

                    val_loss += criterion(logps, labels)

                print(f"Epoch: {epoch+1}/{epochs}")
                print(f"Train Loss: {running_loss/print_every:.3f}")
                print(f"Test Loss: {val_loss/len(valloader):.3f}")
                print(f"Accuracy: {accuracy/len(valloader):.3f}")

                running_loss = 0

                # resetting to training mode.
                model.train()
                
        # calling function to save the model's necessary information
        checkpoint(model, optimizer, save_dir, epoch, class_to_idx, arch)


def test_model(trained_model, testloader, gpu):
    """This function test the trained model in an unseen dataset.

    Args:
        trained_model (torchvision.model): Trained Model.
        testloader: Test set of images
        gpu (bool): Flag for allowing the use of GPU as the device.
    """

    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    for param in trained_model.parameters():
        param.requires_grad = False

    trained_model.eval()
    test_accuracy = 0
    for images, labels in testloader:

        # print(images)
        images, labels = images.to(device), labels.to(device)

        logps = trained_model.forward(images)
        ps = torch.exp(logps)

        top_ps, top_class = ps.topk(1, dim=1)

        equal = top_class == labels.view(*top_class.shape)

        test_accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

    print(f"Test Accuracy: {test_accuracy/len(testloader):.3f}")


@click.command()
@click.argument("data_dir")
@click.option(
    "--learning_rate",
    default=0.003,
    help="Learning Rate for Training Model.",
    type=click.FLOAT,
)
@click.option(
    "--save_dir",
    default="checkpoint.pth",
    help="Path to save model checkpoint.",
    type=click.STRING,
)
@click.option(
    "--arch",
    default="resnet",
    help="Pretrained Architecture (resnet, alexnet or vgg).",
    type=click.STRING,
)
@click.option(
    "--hidden_units",
    default=2048,
    help="Hidden Units of the final layer. (vgg - 25088, resnet - 2048, alexnet - 9216)",
    type=click.INT,
)
@click.option(
    "--epochs", default=5, help="Number of epochs to train the model.", type=click.INT
)
@click.option(
    "--number_of_categories", default=102, help="Number of categories in the data set.", type=click.INT
)
@click.option(
    "--gpu", default=False, help="Allow use of GPU to train the model.", is_flag=True
)
@click.option(
    "--resume_training", default=False, help="Resume Training Flag.", is_flag=True
)
@click.option(
    "--no_test",
    default=False,
    help="Test Flag. Use this if you don't want to see the model's performance in an unseen test set.",
    is_flag=True,
)
def main(
    data_dir,
    learning_rate,
    save_dir,
    arch,
    hidden_units,
    epochs,
    number_of_categories,
    gpu,
    resume_training,
    no_test,
):

    # Load Data
    if not Path(data_dir).exists():
        click.echo("Data directory doesn't exist!")
        raise SystemExit(1)

    trainloader, valloader, testloader = load_data(data_dir)
    class_to_idx = load_data(data_dir, class_to_index=True)

    if resume_training:
        classifier = load_final_layer(load_checkpoint(save_dir)["hidden_layers"])

    else:
        classifier = final_layer(hidden_units=hidden_units, number_of_categories=number_of_categories)

    if arch == "vgg":
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = classifier

    if arch == "resnet":
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = classifier

    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = classifier

        if resume_training:
            model.load_state_dict(load_checkpoint(save_dir)['state_dict'])

    train_model(
        trainloader=trainloader,
        valloader=valloader,
        class_to_idx=class_to_idx,
        save_dir=save_dir,
        model=model,
        arch=arch,
        gpu=gpu,
        epochs=epochs,
        learning_rate=learning_rate,
        resume_training=resume_training,
    )

    if not no_test:
        test_model(trained_model=model, testloader=testloader, gpu=gpu)


if __name__ == "__main__":

    main()
