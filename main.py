from imagenet_c import corrupt
import image_utils
from model_tools import Model_tools
import torch


if __name__ == "__main__":
    mt = Model_tools()
    mt.load_weights("best_model_res50_clean.pt")
    # model_tools.eval_corruption(model, "imagenet_corrupted", 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # image_utils.corrupt_imagenet("sub-imagenet-200", "imagenet_corrupted")
    # image_utils.perturb_imagenet("sub-imagenet-200", "imagenet_perturbed")
    mt.eval_pert("gaussian_noise", "imagenet_perturbed", 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    