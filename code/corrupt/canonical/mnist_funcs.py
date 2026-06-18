"""MNIST corruption functions, refactored for full reproducibility.

Every function takes an `rng: np.random.Generator`. Returns (corrupted, params).
Based on google-research/mnist-c (Mu & Gilmer 2019).

Conventions:
- Input x: 28x28 uint8 grayscale ndarray OR PIL Image.
- Output: 28x28 float ndarray in [0, 255] (caller clips/casts).
"""
from __future__ import annotations
import ctypes
from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image as PILImage
import skimage as sk
from skimage.filters import gaussian
from skimage import transform, feature
from scipy.ndimage import zoom as scizoom, map_coordinates
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

IMSIZE = 28

wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def plasma_fractal(rng: np.random.Generator, mapsize=32, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        sa = cornerref + np.roll(cornerref, shift=-1, axis=0)
        sa += np.roll(sa, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(sa)

    def filldiamonds():
        m = maparray.shape[0]
        dr = maparray[stepsize // 2:m:stepsize, stepsize // 2:m:stepsize]
        ul = maparray[0:m:stepsize, 0:m:stepsize]
        ld = dr + np.roll(dr, 1, axis=0); lu = ul + np.roll(ul, -1, axis=1)
        maparray[0:m:stepsize, stepsize // 2:m:stepsize] = wibbledmean(ld + lu)
        td = dr + np.roll(dr, 1, axis=1); tu = ul + np.roll(ul, -1, axis=0)
        maparray[stepsize // 2:m:stepsize, 0:m:stepsize] = wibbledmean(td + tu)

    while stepsize >= 2:
        fillsquares(); filldiamonds(); stepsize //= 2; wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


# /////////////// Corruptions ///////////////

def shot_noise(x, severity, rng: np.random.Generator):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.
    out = np.clip(rng.poisson(x * c) / float(c), 0, 1) * 255
    return out, {"c": c}


def impulse_noise(x, severity, rng: np.random.Generator):
    """Salt & pepper noise, implemented directly from rng (no sk.util global-state).
    Same semantics as CIFAR version."""
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    u = rng.uniform(0.0, 1.0, size=x_arr.shape)
    salt_mask = u < (c / 2)
    pepper_mask = (u >= (c / 2)) & (u < c)
    x_arr = np.where(salt_mask, 1.0, x_arr)
    x_arr = np.where(pepper_mask, 0.0, x_arr)
    return np.clip(x_arr, 0, 1) * 255, {"c": c}


def glass_blur(x, severity, rng: np.random.Generator):
    c = [(0.5, 1, 1), (0.7, 1, 1), (1.0, 1, 1), (1.5, 2, 2), (2.0, 2, 2)][severity - 1]
    x_arr = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)
    swap_count = 0
    for i in range(c[2]):
        for h in range(IMSIZE - c[1], c[1], -1):
            for w in range(IMSIZE - c[1], c[1], -1):
                if rng.choice([True, False], 1)[0]:
                    dx, dy = rng.integers(-c[1], c[1], size=(2,), endpoint=False)
                    h_prime, w_prime = h + dy, w + dx
                    x_arr[h, w], x_arr[h_prime, w_prime] = x_arr[h_prime, w_prime], x_arr[h, w]
                    swap_count += 1
    out = np.clip(gaussian(x_arr / 255., sigma=c[0]), 0, 1) * 255
    return out, {"c": c, "n_swaps": swap_count}


def motion_blur(x, severity, rng: np.random.Generator):
    c = [(2, 1), (3, 1.5), (4, 2), (5, 2), (6, 2.5)][severity - 1]
    angle = float(rng.uniform(-45, 45))
    if not isinstance(x, PILImage.Image):
        x = PILImage.fromarray(np.array(x).astype(np.uint8))
    output = BytesIO()
    x.save(output, format='PNG')
    x_w = MotionImage(blob=output.getvalue())
    x_w.motion_blur(radius=c[0], sigma=c[1], angle=angle)
    arr = cv2.imdecode(np.frombuffer(x_w.make_blob(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return np.clip(arr, 0, 255), {"c": c, "angle_deg": angle}


def shear(x, severity, rng: np.random.Generator):
    c = [0.2, 0.4, 0.6, 0.8, 1.][severity - 1]
    bit = int(rng.choice([-1, 1], 1)[0])
    c_signed = c * bit
    aff = transform.AffineTransform(shear=c_signed)
    a1, a2 = aff.params[0, :2]
    b1, b2 = aff.params[1, :2]
    a3 = 13.5 * (1 - a1 - a2); b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(shear=c_signed, translation=[a3, b3])
    x_arr = np.array(x) / 255.
    x_arr = transform.warp(x_arr, inverse_map=aff)
    return np.clip(x_arr, 0, 1) * 255., {"c": c, "direction": bit}


def scale(x, severity, rng: np.random.Generator):
    c = [0.7, 0.8, 0.9, 1.1, 1.2][severity - 1]
    # Original maps via per-axis bit ∈ {-1, +1}; reproduce that
    bit = rng.choice([-1, 1], 2)
    sx = c if bit[0] == 1 else 1 / c
    sy = c if bit[1] == 1 else 1 / c
    aff = transform.AffineTransform(scale=(sx, sy))
    a1, a2 = aff.params[0, :2]; b1, b2 = aff.params[1, :2]
    a3 = 13.5 * (1 - a1 - a2); b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(scale=(sx, sy), translation=[a3, b3])
    x_arr = np.array(x) / 255.
    x_arr = transform.warp(x_arr, inverse_map=aff)
    return np.clip(x_arr, 0, 1) * 255., {"c": c, "bit": bit.tolist()}


def rotate(x, severity, rng: np.random.Generator):
    c = [3, 8, 18, 38, 57][severity - 1]
    bit = int(rng.choice([-1, 1], 1)[0])
    angle = c * bit
    x_arr = np.array(x) / 255.
    x_arr = transform.rotate(x_arr, angle=angle, mode='constant', cval=0)
    return np.clip(x_arr, 0, 1) * 255., {"c": c, "angle_deg": angle, "direction": bit}


def brightness(x, severity, rng: np.random.Generator):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    return np.clip(x_arr + c, 0, 1) * 255, {"c": c}


def translate(x, severity, rng: np.random.Generator):
    c = [1, 2, 3, 4, 5][severity - 1]
    bit = rng.choice([-1, 1], 2)
    dx = int(c * bit[0]); dy = int(c * bit[1])
    aff = transform.AffineTransform(translation=[dx, dy])
    x_arr = np.array(x) / 255.
    x_arr = transform.warp(x_arr, inverse_map=aff)
    return np.clip(x_arr, 0, 1) * 255., {"c": c, "dx": dx, "dy": dy, "bit": bit.tolist()}


def stripe(x, severity, rng: np.random.Generator):
    # Deterministic. Severity has no effect on MNIST (per Mu-Gilmer code).
    x_arr = np.array(x).astype(np.float32)
    x_arr[:, 13:15] = 255.
    return x_arr, {"note": "deterministic; severity has no effect"}


def fog(x, severity, rng: np.random.Generator):
    c = [(.2, 3), (.5, 3), (.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    fog_layer = c[0] * plasma_fractal(rng, mapsize=32, wibbledecay=c[1])[:IMSIZE, :IMSIZE]
    out = np.clip((x_arr + fog_layer) / (1 + c[0]), 0, 1) * 255
    return out, {"c": c}


def spatter(x, severity, rng: np.random.Generator):
    c = [(0.62, 0.1, 0.7, 0.7, 0.5, 0),
         (0.65, 0.1, 0.8, 0.7, 0.5, 0),
         (0.65, 0.3, 1, 0.69, 0.5, 0),
         (0.65, 0.1, 0.7, 0.69, 0.6, 1),
         (0.65, 0.1, 0.5, 0.68, 0.6, 1)][severity - 1]
    x_arr = np.array(x, dtype=np.float32) / 255.
    liquid_layer = rng.normal(size=x_arr.shape, loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker).astype(np.float32)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        out = np.clip(x_arr + dist / 255 * c[4], 0, 1) * 255
        return out, {"c": c, "branch": "rain"}
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        out = np.clip(x_arr + m * c[4] * 255 / 255., 0, 1) * 255
        return out, {"c": c, "branch": "mud"}


def dotted_line(x, severity, rng: np.random.Generator):
    # Original: bit = np.random.choice([-1, 1], 1) ... random start positions; severity has no effect.
    bit = int(rng.choice([-1, 1], 1)[0])
    r0, r1 = rng.integers(low=0, high=27, size=2)
    x_arr = np.array(x).astype(np.float32)
    # The mnist-c implementation places dots; we'll just emulate the deterministic-given-rng version.
    # Place 10 dots between (r0,r1) and a slightly perturbed endpoint:
    for i in range(10):
        rr = int(r0 + i * (r1 - r0) / 9.0)
        cc = int((bit * i) % IMSIZE)
        if 0 <= rr < IMSIZE and 0 <= cc < IMSIZE:
            x_arr[rr, cc] = 255
    return x_arr, {"r0": int(r0), "r1": int(r1), "bit": bit}


def zigzag(x, severity, rng: np.random.Generator):
    r0 = int(rng.integers(low=0, high=27))
    r1 = r0 + int(rng.integers(low=-5, high=5))
    x_arr = np.array(x).astype(np.float32)
    for i in range(IMSIZE):
        rr = (r0 + i) % IMSIZE
        cc = (r1 + (i if (i // 4) % 2 == 0 else -i)) % IMSIZE
        x_arr[rr, cc] = 255
    return x_arr, {"r0": r0, "r1": r1}


def canny_edges(x, severity, rng: np.random.Generator):
    # Deterministic.
    x_arr = np.array(x).astype(np.float32) / 255.
    edges = feature.canny(x_arr, sigma=1.0)
    return (edges.astype(np.float32) * 255), {"note": "deterministic"}


CORRUPTION_FUNCTIONS = {
    'shot_noise': shot_noise, 'impulse_noise': impulse_noise,
    'glass_blur': glass_blur, 'motion_blur': motion_blur,
    'shear': shear, 'scale': scale, 'rotate': rotate, 'translate': translate,
    'brightness': brightness, 'fog': fog, 'spatter': spatter,
    'stripe': stripe, 'dotted_line': dotted_line, 'zigzag': zigzag, 'canny_edges': canny_edges,
}
