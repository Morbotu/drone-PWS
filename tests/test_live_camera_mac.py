import pytest
from src.live_camera_mac import live_camera_mac


class TestDownloadModel:
    def test_encoder_path(self):
        assert live_camera_mac.download_model(
            "mono+stereo_640x192")[0] == "src/live_camera_mac/models/mono+stereo_640x192/encoder.pth"

    def test_depth_decoder_path(self):
        assert live_camera_mac.download_model(
            "mono+stereo_640x192")[1] == "src/live_camera_mac/models/mono+stereo_640x192/depth.pth"


@pytest.fixture
def device():
    return live_camera_mac.get_device()


@pytest.fixture
def encoder_path():
    return live_camera_mac.download_model("mono+stereo_640x192")[0]


def test_load_trained_model(encoder_path, device):
    live_camera_mac.load_trained_model(encoder_path, device)
