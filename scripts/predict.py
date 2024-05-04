import json
import torch
import click
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt
from helper import preprocess, load_final_layer, imshow
from training_help import load_checkpoint


def predict(image_path, model, top_k, device):
    """Predict the class (or classes) of an image using a trained deep learning model."""

    model.eval()
    model.to(device)
    image = preprocess(image_path).to(device)

    logps = model.forward(image.float().unsqueeze(0))

    ps = torch.exp(logps)

    top_ps, top_class = ps.topk(top_k, dim=1)

    reversed_idx = {value: key for key, value in model.class_to_idx.items()}

    classes = [reversed_idx.get(val.item()) for val in top_class[0]]

    return top_ps, classes


def sanity_check(image_path, probs, classes, save_path):

    fig, ax = plt.subplots(2, 1, figsize=(4, 6))
    image = preprocess(image_path)
    imshow(image, ax=ax[0])
    ax[1].barh(width=probs[0].numpy()[::-1], y=classes[::-1])
    plt.tight_layout()

    fig.savefig(save_path)

@click.command()
@click.argument("image_path", type=click.STRING)
@click.argument("checkpoint", type=click.STRING)
@click.option(
    "--top_k",
    default=5,
    help="Number of top probabilities predcited by the model.",
    type=click.INT,
)
@click.option(
    "--category_name",
    default="../files/cat_to_name.json",
    help="File with the mapping of categories to real names.",
)
@click.option(
    "--gpu", default=False, help="Flag to allow use of GPU at prediction.", is_flag=True
)
@click.option(
    "--save_path", default='inference_example.jpg', help="Path to store the sanity plot."
)
@click.option(
    "--plot", default=False, is_flag=True, help='Set this if you wish save the sanity check plot as an image.'
)
def main(image_path, checkpoint, top_k, category_name, gpu, save_path, plot):

    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # load model checkpoint parameters
    model_checkpoint = load_checkpoint(checkpoint)
    # Load the final layer structure
    trained_classifier = load_final_layer(model_checkpoint["hidden_layers"])

    # Cat to name mapping
    with open(category_name, "r") as file:
        cat_to_name = json.load(file)

    if model_checkpoint["arch"] == "vgg":
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = trained_classifier

    if model_checkpoint["arch"] == "resnet":
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = trained_classifier

    if model_checkpoint["arch"] == "alexnet":
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = trained_classifier

    model.load_state_dict(model_checkpoint["state_dict"])

    for param in model.parameters():
        param.requires_grad = False

    # loading class_to_idx from checkpoint
    model.class_to_idx = model_checkpoint["class_to_idx"]

    probs, classes = predict(
        image_path=image_path, model=model, top_k=top_k, device=device
    )

    class_names = [cat_to_name[class_id] for class_id in classes]
    print(f"Probabilities: {probs.numpy()[0]}")
    print(f"Classes: {class_names}")

    if plot:
        sanity_check(image_path=image_path, probs=probs, classes=class_names,save_path=save_path)


if __name__ == "__main__":
    main()
