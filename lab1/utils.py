import time
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    arr = np.asarray(img).astype(np.float32)
    return arr


def save_uint8_image(arr: np.ndarray, path: str):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def benchmark(func, img, runs=10):
    for _ in range(2):
        func(img)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(img)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return times.mean(), times.std(), times
