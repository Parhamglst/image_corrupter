from model_tools import Model_tools


if __name__ == "__main__":
    # image_utils.corrupt_imagenet("sub-imagenet-200", "imagenet_corrupted")
    # image_utils.perturb_imagenet("sub-imagenet-200", "imagenet_perturbed")
    mt = Model_tools()
    mt.load_weights("best_model_res50_clean.pt")
    mt.eval_corruption(
        "imagenet_corrupted", 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    mt.eval_pert(
        "tilt", "imagenet_perturbed", 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
