#!/usr/bin/env python3

"""
Dit programma zet de mac camera aan om te filmen en laat het op het scherm zien.

Door Rombout Jansen
"""

# Modules
import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms
import cv2

# Geinstalleerde bestanden
import networks
from utils import download_model_if_doesnt_exist


def main() -> None:
    display_depth()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def display_depth(model_name: str = "mono+stereo_640x192") -> None:
    """Functie die de met de camera filmt, het beeld omzet in diepte en een preview ervan laat zien.
    """

    device = get_device()

    # Download het model
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # Laat getrainde model
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        cv2.namedWindow("preview")
        cv2.namedWindow("normal")
        vc = cv2.VideoCapture(0)

        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
            frame2 = frame
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            cv2.imshow("normal", frame2)
            rval, frame = vc.read()
            frame2 = frame

            # Load image and preprocess
            frame = pil.fromarray(frame)
            original_width, original_height = frame.size
            input_image = frame.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(
                vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[
                              :, :, :3] * 255).astype(np.uint8)
            frame = pil.fromarray(colormapped_im)
            frame = np.asarray(frame)

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

        cv2.destroyWindow("preview")


if __name__ == "__main__":
    main()
