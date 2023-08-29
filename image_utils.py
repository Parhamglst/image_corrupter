from PIL import Image
import numpy as np
from imagenet_c import corrupt
from pathlib import Path
from os import path, makedirs

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

def _corrupt_image(image, corruption_name, severity, output_path):
    img_arr = np.asarray(Image.open(image))
    corrupted_img_arr = corrupt(img_arr, corruption_name=corruption_name, severity=severity)
    corrupted_image = Image.fromarray(corrupted_img_arr)
    makedirs(output_path, exist_ok=True)
    out = path.join(output_path, "{}-{}.jpg".format(image.stem, corruption_name))
    corrupted_image.save(out, "JPEG")
    
def corrupt_imagenet(imagenet_folder, output_folder):
    fdir = Path(imagenet_folder + "/test")
    for subdir in fdir.iterdir():
        if subdir.is_dir():
            for img in subdir.iterdir():
                if img.is_file():
                    for corruption in corruptions:
                        _corrupt_image(img, corruption, 1, output_folder)