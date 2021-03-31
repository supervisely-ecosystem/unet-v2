import os
from pathlib import Path
import sys
import tarfile
import json
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.functional import softmax
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
        model_parallel = DataParallel(model)
        model_parallel.load_state_dict(weights)
        torch.save(model_parallel.module.state_dict(), dst_path)


def to_model_input(model_config, image):
    input_width = model_config["settings"]["input_size"]["width"]
    input_height = model_config["settings"]["input_size"]["height"]
    input_size = (input_width, input_height)
    resized_image = cv2.resize(image, input_size)

    input_image_normalizer = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    raw_input = input_image_normalizer(resized_image)

    # fake batch dimension required to fit network's input dimensions
    # model_input = torch.stack([raw_input], 0)  # add dim #0 (batch size 1)
    model_input = raw_input.unsqueeze(0)
    return model_input


def predict(device, model, model_config, image):
    original_height, original_width = image.shape[:2]

    model_input = to_model_input(model_config, image)
    model_input = model_input.to(device, torch.float)
    output = model(model_input)
    output = softmax(output, dim=1)

    output = output.data.cpu().numpy()[0]  # from batch to 3d
    pred = np.transpose(output, (1, 2, 0))
    pred = cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

    results = {}
    pred_classes = np.argmax(pred, axis=2)
    for class_name, class_index in model_config["class_title_to_idx"].items():
        predicted_class_pixels = (pred_classes == class_index)
        mask = predicted_class_pixels.astype(np.uint8) * 255
        results[class_name] = mask
    return results


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

    # use one of:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cuda:0')
    device = torch.device('cpu')

    # load model input resolution and the list of output classes
    model_config_path = os.path.join(weights_dir, "config.json")
    with open(model_config_path) as f:
        model_config = json.load(f)
    num_output_classes = len(model_config["class_title_to_idx"])

    # construct model graph
    model = construct_unet(n_cls=num_output_classes)

    weights_path = os.path.join(weights_dir, "model.pt")
    #load model as DataParallel, can be used only on GPU
    #model = DataParallel(model).cuda()
    #model.load_state_dict(torch.load(weights_path, map_location=device))

    # convert weights from DataParallel to generic format
    generic_weights_path = os.path.join(weights_dir, "generic_model.pt")
    if sly.fs.file_exists(generic_weights_path) is False:
        convert_weights_to_generic_format(model, weights_path, generic_weights_path)
    # model can be used both on CPU or GPU
    model = construct_unet(n_cls=num_output_classes)
    model.load_state_dict(torch.load(generic_weights_path, map_location=device))

    # model to device and and set to inference mode
    model.to(device)
    model.eval()

    # inference on image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = predict(device, model, model_config, image)
    for class_name, class_mask in results.items():
        cv2.imwrite(os.path.join(data_dir, f"{class_name}.png"), class_mask)

    # convert to ONNX format
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    onnx_weights_path = os.path.join(weights_dir, "model.onnx")
    inp = to_model_input(model_config, image)
    torch.onnx.export(model,
                      inp,
                      onnx_weights_path,
                      opset_version=11,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}}
                      )

    # verify onnx model
    import onnx
    onnx_model = onnx.load(onnx_weights_path)
    onnx.checker.check_model(onnx_model)

    # verify onnx model


if __name__ == "__main__":
    main()