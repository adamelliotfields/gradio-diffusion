from threading import Lock
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _to_hwc3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    height, width, channels = image.shape
    if channels == 3:
        return image
    if channels == 1:
        return np.concatenate([image, image, image], axis=2)
    if channels == 4:
        color = image[:, :, 0:3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        blended = color * alpha + 255.0 * (1.0 - alpha)
        return blended.clip(0, 255).astype(np.uint8)
    raise ValueError(
        f"Invalid image shape ({height}, {width}, {channels}); expected 1, 3, or 4 channels"
    )


def _resize_to_detection_resolution(image: np.ndarray, resolution: int) -> np.ndarray:
    height, width, _ = image.shape
    scale = float(resolution) / float(min(height, width))
    height = int(np.round((height * scale) / 64.0)) * 64
    width = int(np.round((width * scale) / 64.0)) * 64
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


class CannyAnnotator:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        resolution = min(*size)
        image = np.array(img, dtype=np.uint8)
        image = _to_hwc3(image)
        image = _resize_to_detection_resolution(image, resolution)
        edges = cv2.Canny(image, 100, 200)
        edges = _to_hwc3(edges)
        edges = cv2.resize(edges, size, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(edges)
