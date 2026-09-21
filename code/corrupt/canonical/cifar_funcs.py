"""CIFAR-10 corruption functions, refactored for full reproducibility.

Every function takes an `rng: np.random.Generator` parameter. No function uses
the global np.random state. Each function returns (corrupted_image, params_dict)
where params_dict records the sampled values for reproducibility verification.

Based on hendrycks/robustness (Hendrycks & Dietterich 2019).

Conventions:
- Input x: 32x32x3 uint8 RGB ndarray OR PIL Image (functions that need PIL convert internally).
- Output: 32x32x3 uint8-cast float ndarray, range [0, 255].
- rng: np.random.Generator (used instead of np.random.* globals).
- params: dict with corruption-specific sampled values + a 'rng_seed' field.
"""
from __future__ import annotations
import ctypes
from io import BytesIO
import os
import warnings

import cv2
import numpy as np
from PIL import Image as PILImage
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom, map_coordinates
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

IMSIZE = 32

wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# /////////////// Helpers ///////////////

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(rng: np.random.Generator, mapsize=32, wibbledecay=3):
    """Generate plasma-fractal cloud (for fog corruption). Stochastic via rng."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize2 = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize2:stepsize, stepsize // 2:mapsize2:stepsize]
        ulgrid = maparray[0:mapsize2:stepsize, 0:mapsize2:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize2:stepsize, stepsize // 2:mapsize2:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize2:stepsize, 0:mapsize2:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// Corruptions ///////////////

def gaussian_noise(x, severity, rng: np.random.Generator):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.
    out = np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255
    return out, {"c": c}


def shot_noise(x, severity, rng: np.random.Generator):
    c = [500, 250, 100, 75, 50][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.
    out = np.clip(rng.poisson(x * c) / c, 0, 1) * 255
    return out, {"c": c}


def impulse_noise(x, severity, rng: np.random.Generator):
    """Salt & pepper noise, implemented directly from rng (no sk.util global-state).
    Matches sk.util.random_noise(mode='s&p', amount=c, salt_vs_pepper=0.5) semantics:
        - Each pixel independently gets corrupted with probability c.
        - Of corrupted pixels, half become 1.0 (salt), half become 0.0 (pepper).
    """
    c = [.01, .02, .03, .05, .07][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    # Per-pixel uniform draws over the full image shape
    u = rng.uniform(0.0, 1.0, size=x_arr.shape)
    # u < c/2  -> salt (=1.0);  u < c -> pepper (=0.0);  else unchanged.
    salt_mask = u < (c / 2)
    pepper_mask = (u >= (c / 2)) & (u < c)
    x_arr = np.where(salt_mask, 1.0, x_arr)
    x_arr = np.where(pepper_mask, 0.0, x_arr)
    out = np.clip(x_arr, 0, 1) * 255
    return out, {"c": c}


def defocus_blur(x, severity, rng: np.random.Generator):
    # Fully deterministic given x, severity.
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    out = np.clip(channels, 0, 1) * 255
    return out, {"c": c}


def glass_blur(x, severity, rng: np.random.Generator):
    c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][severity - 1]
    x_arr = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    swaps = []
    for i in range(c[2]):
        for h in range(IMSIZE - c[1], c[1], -1):
            for w in range(IMSIZE - c[1], c[1], -1):
                dx, dy = rng.integers(-c[1], c[1], size=(2,), endpoint=False)
                # Note: original uses np.random.randint(low, high) which is [low, high)
                # rng.integers default is [low, high); we add endpoint=False explicitly for clarity.
                swaps.append((int(h), int(w), int(dx), int(dy)))
                h_prime, w_prime = h + dy, w + dx
                x_arr[h, w], x_arr[h_prime, w_prime] = x_arr[h_prime, w_prime].copy(), x_arr[h, w].copy()
    out = np.clip(gaussian(x_arr / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255
    return out, {"c": c, "n_swaps": len(swaps)}  # Don't log every swap (would explode params.jsonl)


def motion_blur(x, severity, rng: np.random.Generator):
    c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]
    angle = float(rng.uniform(-45, 45))
    if not isinstance(x, PILImage.Image):
        x = PILImage.fromarray(np.array(x).astype(np.uint8))
    output = BytesIO()
    x.save(output, format='PNG')
    x_w = MotionImage(blob=output.getvalue())
    x_w.motion_blur(radius=c[0], sigma=c[1], angle=angle)
    x_arr = cv2.imdecode(np.frombuffer(x_w.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    if x_arr.shape == (IMSIZE, IMSIZE):
        return np.clip(x_arr[..., [2, 1, 0]], 0, 255), {"c": c, "angle_deg": angle}  # B&W
    out = np.clip(x_arr[..., [2, 1, 0]], 0, 255)  # BGR -> RGB
    return out, {"c": c, "angle_deg": angle}


def zoom_blur(x, severity, rng: np.random.Generator):
    # Fully deterministic given x, severity.
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    x_arr = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x_arr)
    for zoom_factor in c:
        out += clipped_zoom(x_arr, zoom_factor)
    x_arr = (x_arr + out) / (len(c) + 1)
    return np.clip(x_arr, 0, 1) * 255, {"c": [float(z) for z in c]}


def fog(x, severity, rng: np.random.Generator):
    c = [(.2, 3), (.5, 3), (.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    max_val = x_arr.max()
    x_arr += c[0] * plasma_fractal(rng, wibbledecay=c[1])[:IMSIZE, :IMSIZE][..., np.newaxis]
    out = np.clip(x_arr * max_val / (max_val + c[0]), 0, 1) * 255
    return out, {"c": c}


def frost(x, severity, rng: np.random.Generator, frost_dir: str = None):
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    if frost_dir is None:
        frost_dir = os.path.join(os.path.dirname(__file__), 'data', 'frost_images')
    idx = int(rng.integers(0, 5))
    filename = [os.path.join(frost_dir, f) for f in
                ['frost1.png', 'frost2.png', 'frost3.png',
                 'frost4.jpg', 'frost5.jpg', 'frost6.jpg']][idx]
    frost_img = cv2.imread(filename)
    if frost_img is None:
        raise FileNotFoundError(f"frost image {filename} not found")
    x_start = int(rng.integers(0, frost_img.shape[0] - IMSIZE))
    y_start = int(rng.integers(0, frost_img.shape[1] - IMSIZE))
    frost_img = frost_img[x_start:x_start + IMSIZE, y_start:y_start + IMSIZE][..., [2, 1, 0]]
    out = np.clip(c[0] * np.array(x) + c[1] * frost_img, 0, 255)
    return out, {"c": c, "frost_idx": idx, "crop_xy": [x_start, y_start]}


def snow(x, severity, rng: np.random.Generator):
    c = [(0.1, 0.2, 1, 0.6, 8, 3, 0.95),
         (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
         (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
         (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
         (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8)][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    snow_layer = rng.normal(size=x_arr.shape[:2], loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer_pil = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer_pil.save(output, format='PNG')
    snow_w = MotionImage(blob=output.getvalue())
    angle = float(rng.uniform(-135, -45))
    snow_w.motion_blur(radius=c[4], sigma=c[5], angle=angle)
    snow_arr = cv2.imdecode(np.frombuffer(snow_w.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_arr = snow_arr[..., np.newaxis]
    x_arr = c[6] * x_arr + (1 - c[6]) * np.maximum(
        x_arr, cv2.cvtColor((x_arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).reshape(IMSIZE, IMSIZE, 1) / 255. * 1.5 + 0.5)
    out = np.clip(x_arr + snow_arr + np.rot90(snow_arr, k=2), 0, 1) * 255
    return out, {"c": c, "angle_deg": angle}


def brightness(x, severity, rng: np.random.Generator):
    # Fully deterministic given x, severity.
    c = [.05, .1, .15, .2, .3][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    x_arr = sk.color.rgb2hsv(x_arr)
    x_arr[:, :, 2] = np.clip(x_arr[:, :, 2] + c, 0, 1)
    x_arr = sk.color.hsv2rgb(x_arr)
    return np.clip(x_arr, 0, 1) * 255, {"c": c}


def contrast(x, severity, rng: np.random.Generator):
    # Fully deterministic.
    c = [.75, .5, .4, .3, 0.15][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    means = np.mean(x_arr, axis=(0, 1), keepdims=True)
    return np.clip((x_arr - means) * c + means, 0, 1) * 255, {"c": c}


def elastic_transform(image, severity, rng: np.random.Generator):
    IMAGE_SIZE = IMSIZE
    c = [(IMAGE_SIZE * 0, IMAGE_SIZE * 0, IMAGE_SIZE * 0.08),
         (IMAGE_SIZE * 0.05, IMAGE_SIZE * 0.2, IMAGE_SIZE * 0.07),
         (IMAGE_SIZE * 0.08, IMAGE_SIZE * 0.06, IMAGE_SIZE * 0.06),
         (IMAGE_SIZE * 0.1, IMAGE_SIZE * 0.04, IMAGE_SIZE * 0.05),
         (IMAGE_SIZE * 0.1, IMAGE_SIZE * 0.03, IMAGE_SIZE * 0.03)][severity - 1]
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2_offset = rng.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    pts2 = pts1 + pts2_offset
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (gaussian(rng.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(rng.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1)))
    out = np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255
    return out, {"c": c, "pts2_offset_shape": list(pts2_offset.shape)}


def pixelate(x, severity, rng: np.random.Generator):
    # Fully deterministic.
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    if not isinstance(x, PILImage.Image):
        x = PILImage.fromarray(np.array(x).astype(np.uint8))
    x = x.resize((int(IMSIZE * c), int(IMSIZE * c)), PILImage.BOX)
    x = x.resize((IMSIZE, IMSIZE), PILImage.BOX)
    return np.array(x), {"c": c}


def jpeg_compression(x, severity, rng: np.random.Generator):
    # Fully deterministic.
    c = [80, 65, 58, 50, 40][severity - 1]
    if not isinstance(x, PILImage.Image):
        x = PILImage.fromarray(np.array(x).astype(np.uint8))
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    return np.array(PILImage.open(output)), {"c": c}


CORRUPTION_FUNCTIONS = {
    'gaussian_noise': gaussian_noise, 'shot_noise': shot_noise, 'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur, 'glass_blur': glass_blur, 'motion_blur': motion_blur, 'zoom_blur': zoom_blur,
    'snow': snow, 'frost': frost, 'fog': fog, 'brightness': brightness,
    'contrast': contrast, 'elastic_transform': elastic_transform,
    'pixelate': pixelate, 'jpeg_compression': jpeg_compression,
}
