#!/usr/bin/env python3

"""This program films and converts live video to depth view.

By Rombout Jansen
"""

# Typing
from typing import Any, OrderedDict

# Modules
import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torch.functional import Tensor
from torch.serialization import load
from torchvision import transforms
import cv2

# Downloaded files
from utils import download_model_if_doesnt_exist
import networks


def get_device() -> torch.device:
    """Opens a device

    Returns:
        torch.device: The device used to analyze the pictures
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def download_model(model_name: str) -> tuple[str, str]:
    """Downloads the model

    Args:
        model_name (str): The name of the model

    Returns:
        tuple[str, str]: The paths to the depth decoder and the encoder
    """
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("src/live_camera_mac/models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    return encoder_path, depth_decoder_path


def load_trained_model(encoder_path: str,
                       device: torch.device) -> tuple[networks.ResnetEncoder, dict]:
    """Load the trained model

    Args:
        encoder_path (str): The path to the encoder
        device (torch.device): The device 

    Returns:
        tuple[networks.ResnetEncoder, dict]: The ResnetEncoder and the data
    """
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    return encoder, loaded_dict_enc


def extract_with__and_height(loaded_dict_enc: dict,
                             encoder: networks.ResnetEncoder,
                             device: torch.device) -> tuple[int, int, networks.ResnetEncoder]:
    """Gets the height and the with of the model

    Args:
        loaded_dict_enc (dict): The data loaded
        encoder (networks.ResnetEncoder): The encoder
        device (torch.device): The device used

    Returns:
        tuple[int, int, networks.ResnetEncoder]: The width, height and the encoder
    """
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = OrderedDict({
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    })
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    return feed_height, feed_width, encoder


def load_pretrained_decoder(encoder: networks.ResnetEncoder,
                            depth_decoder_path: str,
                            device: torch.device) -> networks.DepthDecoder:
    """Loads the pretrained model

    Args:
        encoder (networks.ResnetEncoder): The encoder
        depth_decoder_path (str): The path to the depth decoder
        device (torch.device): The device used

    Returns:
        networks.DepthDecoder: The depth decoder
    """
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    return depth_decoder


def preproccess_image(frame: Any,
                      feed_width: int,
                      feed_height: int) -> tuple[int, int, transforms.Tensor]:
    """Preprocesses the image

    Args:
        frame (Any): The frame that is being analyzed
        feed_width (int): The width of the frame
        feed_height (int): The height of the frame

    Returns:
        tuple[int, int, transforms.Tensor]: The original width and height and the input image
    """
    frame = pil.fromarray(frame)
    original_width, original_height = frame.size
    input_image = frame.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    return original_width, original_height, input_image


def predict_depth(input_image: transforms.Tensor,
                  device: torch.device,
                  encoder: networks.ResnetEncoder,
                  depth_decoder: networks.DepthDecoder,
                  original_height: int,
                  original_width: int) -> Any:
    """Predicts the depth of a image

    Args:
        input_image (transforms.Tensor): The input frame
        device (torch.device): The device used
        encoder (networks.ResnetEncoder): The ResnetEncoder
        depth_decoder (networks.DepthDecoder): The depth decoder
        original_height (int): The original width
        original_width (int): The original height

    Returns:
        Any: The depth of the image
    """
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(  # type: ignore
        disp, (original_height, original_width), mode="bilinear", align_corners=False)
    return disp_resized


def create_colormap_depth_image(disp_resized: Any) -> np.ndarray:
    """Creates a colormap of the predicted depth

    Args:
        disp_resized (Any): The depth of to image

    Returns:
        np.ndarray: The new image as a np ndarray
    """
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(  # type: ignore
        vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[
        :, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def display_depth(model_name: str) -> None:
    """Films and show a preview of the depth

    Args:
        model_name (str): The name of the model used

    Raises:
        SystemError: If no camera is available
    """
    device = get_device()
    encoder_path, depth_decoder_path = download_model(model_name)
    encoder, loaded_dict_enc = load_trained_model(encoder_path, device)
    feed_height, feed_width, encoder = extract_with__and_height(
        loaded_dict_enc, encoder, device)
    depth_decoder = load_pretrained_decoder(
        encoder, depth_decoder_path, device)

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        cv2.namedWindow("preview")
        cv2.namedWindow("normal")
        vc = cv2.VideoCapture(0)

        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
            raise SystemError("Not able to open camera")

        while rval:
            rval, frame = vc.read()

            original_width, original_height, input_image = preproccess_image(
                frame, feed_width, feed_height)
            disp_resized = predict_depth(
                input_image, device, encoder, depth_decoder, original_height, original_width)
            colormapped_im = create_colormap_depth_image(disp_resized)
            pil_frame = pil.fromarray(colormapped_im)
            frame_array = np.asarray(pil_frame)
            cv2.imshow("preview", frame_array)
            cv2.imshow("normal", frame)

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

        cv2.destroyWindow("preview")


if __name__ == "__main__":
    display_depth("mono+stereo_640x192")
