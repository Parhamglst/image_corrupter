import torch
import torch.utils.data as data
from resnet import resnet50
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.autograd.variable as V
from image_utils import distortions


def load_weights(path_to_model):
    """this function loads the weights of a resnet50 model

    Args:
        path_to_model (str): path to the model weights

    Returns:
        ResNet50 object: resnet50 model with the weights loaded
    """    
    model = resnet50(pretrained=False)
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


def eval_corruption(model, path_to_corrupted_dataset, batch_size, mean, std):###########################
    """this function evaluates model on a corrupted dataset

    Args:
        model (ResNet50 object): resnet50 model
        path_to_corrupted_dataset (str): path to the corrupted dataset

    Returns:
        
    """
    model.eval()
    
    cudnn.benchmark = True
    
    data.DataLoader(datasets.ImageFolder(root = path_to_corrupted_dataset, 
                                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])),
                    batch_size=batch_size, shuffle= False,pin_memory=True) #########################################################
    
    error_rates = []
    for distortion_name in distortions:
        rate = show_performance(model, distortion_name, path_to_corrupted_dataset, batch_size, mean, std)
        error_rates.append(rate)
        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))


    print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))



def show_performance(model, distortion_name, path_to_corrupted_dataset, batch_size, mean, std):
    """copied from @hendrycks github repo: https://github.com/hendrycks/robustness
    calculates the average error of a model on a given distortion

    Args:
        distortion_name (str): name of the distortion

    Returns:
        float: average error
    """
    errs = []

    for severity in range(1, 6):
        distorted_dataset = datasets.ImageFolder(
            root=path_to_corrupted_dataset + distortion_name + '/' + str(severity),
            transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            data = V(data.cuda(), volatile=True)    ######################

            output = model(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()

        errs.append(1 - 1.*correct / len(distorted_dataset))

    print('\n=Average', tuple(errs))
    return np.mean(errs)
    