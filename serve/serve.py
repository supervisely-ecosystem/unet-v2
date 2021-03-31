import os
from pathlib import Path
import sys
import tarfile
import json
import torch
from torch.nn import DataParallel
from torchvision.transforms import ToTensor, Normalize, Compose
import cv2
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


def forward_model(model, model_config, image, apply_softmax=True):
    original_height, original_width = image.shape[:2]

    input_width = model_config["settings"]["input_size"]["width"]
    input_height = model_config["settings"]["input_size"]["height"]
    input_size = (input_width, input_height)
    resized_image = cv2.resize(image, input_size)

    input_image_normalizer = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    raw_input = input_image_normalizer(resized_image)


    out_shape = None

    model_input = torch.stack([raw_input], 0)  # add dim #0 (batch size 1)
    model_input = cuda_variable(model_input, volatile=True)

    output = model(model_input)
    if apply_softmax:
        output = torch_functional.softmax(output, dim=1)
    output = output.data.cpu().numpy()[0]  # from batch to 3d

    pred = np.transpose(output, (1, 2, 0))
    return sly_image.resize(pred, out_shape)


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

    # load model input resolution and the list of output classes
    model_config_path = os.path.join(weights_dir, "config.json")
    with open(model_config_path) as f:
        model_config = json.load(f)

    # inference on image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




    # resized_img = cv2.resize(img, self.input_size[::-1])
    # model_input = input_image_normalizer(resized_img)
    # # sum(pixelwise_probas_array[0, 0, :]) == 1
    # pixelwise_probas_array = pytorch_inference.infer_per_pixel_scores_single_image(
    #     self.model, model_input, img.shape[:2])
    # labels = raw_to_labels.segmentation_array_to_sly_bitmaps(
    #     self.out_class_mapping, np.argmax(pixelwise_probas_array, axis=2))
    # pixelwise_scores_labels = raw_to_labels.segmentation_scores_to_per_class_labels(
    #     self.out_class_mapping, pixelwise_probas_array)
    # return Annotation(ann.img_size, labels=labels, pixelwise_scores_labels=pixelwise_scores_labels)

if __name__ == "__main__":
    main()