import os
from pathlib import Path
import sys
import tarfile

import torch
from torch.nn import DataParallel

import supervisely_lib as sly

root_source_path = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

data_dir = os.path.join(root_source_path, "data")

from unet import construct_unet
from utils import tree


def download_demo():
    image_url = "https://github.com/supervisely-ecosystem/unet-v2/releases/download/v0.1/demo-lemon-kiwi.jpeg"
    image_path = os.path.join(data_dir, sly.fs.get_file_name_with_ext(image_url))
    if sly.fs.file_exists(image_path) is False:
        sly.fs.download(image_url, image_path)

    weights_tar_url = "https://github.com/supervisely-ecosystem/unet-v2/releases/download/v0.1/unet-lemon-kiwi.tar"
    weights_path = os.path.join(data_dir, sly.fs.get_file_name_with_ext(weights_tar_url))
    if sly.fs.file_exists(weights_path) is False:
        progress = sly.Progress("Downloading weights", 1, is_size=True)
        sly.fs.download(weights_tar_url, weights_path, progress=progress)

    weights_dir = os.path.join(data_dir, sly.fs.get_file_name(weights_tar_url))
    if sly.fs.dir_exists(weights_dir) is False:
        with tarfile.open(weights_path) as archive:
            archive.extractall(weights_dir)

    print("Weights and demo image are downloaded")
    return image_path, weights_dir


# UNet plugin saves model weights as DataParallel
# if you want to use model not only on GPU but also on CPU => convert weights
# https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html#saving-torch-nn-dataparallel-models
def convert_weights_to_generic_format(model, src_path, dst_path):
    weights = torch.load(src_path)
    if list(weights.keys())[0].startswith('module.'):
        model_parallel = DataParallel(model).cuda()
        torch.save(model_parallel.module.state_dict(), dst_path)


def main():
    image_path, weights_dir = download_demo()

    # Example of weights directory structure
    """
     dir: ./data/unet-lemon-kiwi
     ├── model.pt
     └── config.json
    """
    print(f"Weights dir: {weights_dir}")
    for line in tree(Path(weights_dir)):
        print(line)

    model = construct_unet(n_cls=3)

    # convert weights from DataParallel to generic format
    weights_path = os.path.join(weights_dir, "model.pt")
    generic_weights_path = os.path.join(weights_dir, "generic_model.pt")
    if sly.fs.file_exists(generic_weights_path) is False:
        convert_weights_to_generic_format(model, weights_path, generic_weights_path)

    # use one of:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cuda:0')
    device = torch.device('cpu')

    model.load_state_dict(torch.load(generic_weights_path, map_location=device))


if __name__ == "__main__":
    main()