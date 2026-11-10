"""Tiny image helpers (no imaging library needed): spectrograms as PNG."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def png_gray(gray: np.ndarray) -> bytes:
    """Encode an (h, w) uint8 array as an 8 bit greyscale PNG."""
    h, w = gray.shape
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def spectrogram_png(db: np.ndarray, floor: float, top: float) -> bytes:
    """dB spectrogram (frames, bands) -> greyscale PNG.

    One pixel per frame and band, time left to right, low frequencies at the
    bottom, 0 = floor and 255 = top.  The page colours it with a single-hue
    ramp that suits its light or dark theme.
    """
    norm = (np.asarray(db, dtype=np.float64) - floor) / max(top - floor, 1e-9)
    img = np.clip(norm.T[::-1] * 255.0, 0, 255).astype(np.uint8)
    return png_gray(np.ascontiguousarray(img))
