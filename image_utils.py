# downloaded from https://github.com/hendrycks/robustness/blob/master/ImageNet-P/create_p/make_imagenet_p.py by https://github.com/hendrycks

from PIL import Image
import numpy as np
from imagenet_c import corrupt
from pathlib import Path
from os import path, makedirs, getcwd
import make_imagenet_p as mkp

corruptions = {
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
}

perturbations = {
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "zoom_blur",
    "snow",
    "brightness",
    "translate",
    "rotate",
    "tilt",
    "scale",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "shear",
}


def _corrupt_image(image, corruption_name, *output_path) -> None:
    """private function to corrupt an image and save it to a folder

    Args:
        image (posixpath): path of the image to be corrupted
        corruption_name (str): name of the corruption
        severity (int): corruption severity
        output_path (str): path to save the corrupted image
    """
    img_arr = np.asarray(Image.open(image))
    for severity in range(1, 6):
        corrupted_img_arr = corrupt(
            img_arr, corruption_name=corruption_name, severity=severity
        )
        corrupted_image = Image.fromarray(corrupted_img_arr)
        makedirs(
            output_path[0]
            + "/{}/{}/{}".format(corruption_name, str(severity), output_path[1]),
            exist_ok=True,
        )
        out = path.join(
            output_path[0],
            corruption_name,
            str(severity),
            output_path[1],
            "{}.jpg".format(image.stem),
        )
        corrupted_image.save(out, "JPEG")


def corrupt_imagenet(imagenet_folder, output_folder) -> None:
    """this function iterates through every direcdtory in imagenet_folder and corrupts every image in it.
    NOTE: this function does not check trhe subdirectories of imagenet_folder

    Args:
        imagenet_folder (str): path to the imagenet folder
        output_folder (str): path to save the corrupted images
    """
    fdir = Path(imagenet_folder + "/test")
    for subdir in fdir.iterdir():
        if subdir.is_dir():
            for img in subdir.iterdir():
                if img.is_file():
                    for corruption in corruptions:
                        _corrupt_image(img, corruption, output_folder, subdir.name)


def perturb_imagenet(imagenet_folder, output_folder) -> None:
    ip = mkp.Imagenet_p(imagenet_folder + "/test/", output_folder)
    for perturbation in perturbations:
        ip.perturb(perturbation)
