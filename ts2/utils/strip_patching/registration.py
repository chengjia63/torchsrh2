"""Minimal FFT-based registration helpers for standalone strip patching."""

import logging
import math

import numpy as np
import numpy
import scipy.ndimage.interpolation as ndii
from numpy.fft import fft2, ifft2, fftshift

logger = logging.getLogger(__name__)


def highpass(shape):
    assert len(shape) == 2, f"highpass expects a 2D shape, got {shape}"
    x = numpy.outer(
        numpy.cos(numpy.linspace(-math.pi / 2.0, math.pi / 2.0, shape[0])),
        numpy.cos(numpy.linspace(-math.pi / 2.0, math.pi / 2.0, shape[1])),
    )
    return (1.0 - x) * (2.0 - x)


def logpolar(image, angles=None, radii=None):
    assert image.ndim == 2, f"logpolar expects a 2D image, got shape {image.shape}"
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = numpy.empty((angles, radii), dtype="float64")
    theta.T[:] = numpy.linspace(0, numpy.pi, angles, endpoint=False) * -1.0
    distance = numpy.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(distance) / radii)
    radius = numpy.empty_like(theta)
    radius[:] = numpy.power(log_base, numpy.arange(radii, dtype="float64")) - 1.0
    x = radius * numpy.sin(theta) + center[0]
    y = radius * numpy.cos(theta) + center[1]
    output = numpy.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def similarity(im0, im1):
    logger.debug("Running FFT similarity registration")
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    if len(im0.shape) != 2:
        raise ValueError("Images must be 2 dimensional.")

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base**i1

    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base**i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    im2 = ndii.zoom(im1, 1.0 / scale)
    im2 = ndii.rotate(im2, angle)

    if im2.shape < im0.shape:
        tmp = numpy.zeros_like(im0)
        tmp[: im2.shape[0], : im2.shape[1]] = im2
        im2 = tmp
    elif im2.shape > im0.shape:
        im2 = im2[: im0.shape[0], : im0.shape[1]]

    f0 = fft2(im0)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    im2 = ndii.shift(im2, [t0, t1])
    return im2, scale, angle, [-t0, -t1]


def fft_register(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    assert (
        image1.shape == image2.shape
    ), "Input images for registration must have identical shapes."
    assert image1.ndim == 2, f"Expected 2D input images, got shape {image1.shape}"
    logger.debug("Registering strip pair with shape %s", image1.shape)
    registered_image = np.zeros((image1.shape[0], image1.shape[1], 2), dtype=float)
    image3, _, _, _ = similarity(image1, image2)
    registered_image[:, :, 0] = image1
    registered_image[:, :, 1] = image3
    return registered_image
