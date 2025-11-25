import numpy as np
import cv2


def sobel_opencv(img: np.ndarray) -> np.ndarray:
    src = img.astype(np.float32)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = 255.0 * (mag / (mag.max() + 1e-12))
    return mag


Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
Gy = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32)


def pad_reflect(img: np.ndarray, pad: int = 1) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')


def sobel_native_loops(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    src = pad_reflect(img, 1)
    out_x = np.zeros_like(img, dtype=np.float32)
    out_y = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = src[i:i+3, j:j+3]
            out_x[i, j] = np.sum(region * Gx)
            out_y[i, j] = np.sum(region * Gy)

    mag = np.sqrt(out_x**2 + out_y**2)
    mag = 255.0 * (mag / (mag.max() + 1e-12))
    return mag


def sobel_native_vectorized(img: np.ndarray) -> np.ndarray:
    src = pad_reflect(img, 1)
    h, w = img.shape
    acc_x = np.zeros((h, w), dtype=np.float32)
    acc_y = np.zeros((h, w), dtype=np.float32)

    shifts = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1), ( 0, 0), ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

    kernels_x = Gx.flatten()
    kernels_y = Gy.flatten()

    idx = 0
    for dy, dx in shifts:
        block = src[1+dy:1+dy+h, 1+dx:1+dx+w]
        acc_x += kernels_x[idx] * block
        acc_y += kernels_y[idx] * block
        idx += 1

    mag = np.hypot(acc_x, acc_y)
    mag = 255.0 * (mag / (mag.max() + 1e-12))
    return mag
