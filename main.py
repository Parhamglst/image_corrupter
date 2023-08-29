from imagenet_c import corrupt
import image_utils
import model_tools
import torch


if __name__ == "__main__":
    model = model_tools.load_weights("best_model_res50_clean.pt")
    image_utils.corrupt_imagenet("sub-imagenet-200", "imagenet_corrupted")
    