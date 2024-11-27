from threading import Lock
from typing import Tuple

from controlnet_aux import CannyDetector
from PIL import Image


class CannyAnnotator:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.model = CannyDetector()
        return cls._instance

    def __call__(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        resolution = min(*size)
        return self.model(
            img,
            low_threshold=100,
            high_threshold=200,
            detect_resolution=resolution,
            image_resolution=resolution,
        )
