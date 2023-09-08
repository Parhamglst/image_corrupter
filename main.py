from imagenet_c import corrupt
import image_utils
import model_tools
import torch


if __name__ == "__main__":
    model = model_tools.load_weights("best_model_res50_clean.pt")
    model_tools.eval_corruption(model, "imagenet_corrupted", 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image_utils.corrupt_imagenet("sub-imagenet-200", "imagenet_corrupted")
    