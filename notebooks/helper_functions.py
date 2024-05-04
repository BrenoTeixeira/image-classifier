
import torch
from torch import nn 

def final_layer(hidden_layers):

    classifier = nn.Sequential(

        *hidden_layers
    )

    return classifier


def checkpoint(model, optimizer, filename, epoch, class_to_idx, arch):

    checkpoint = {
        'epoch': epoch + 1,
        "arch": arch,
        'output_size': 102,
        'hidden_layers': [each for each in model.fc.children()],
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    
    torch.save(checkpoint, filename)



def load_checkpoint(filename):

    return torch.load(filename)