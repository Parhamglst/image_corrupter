import torch
from torchvision import models

def load_weights(path_to_model):
    """this function loads the weights of a resnet50 model

    Args:
        path_to_model (str): path to the model weights

    Returns:
        ResNet50 object: resnet50 model with the weights loaded
    """    
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(path_to_model))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model.to(device)
    
    model.eval()
    return model
       