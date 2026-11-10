"""Tiny image helpers (no imaging library needed): spectrograms as PNG."""

from __future__ import annotations

import struct
import zlib

import numpy as np

# anchors of a dark-to-bright colour map (perceptually ordered, readable on dark and light pages)
_ANCHORS = np.array([
    [0.00, 8, 10, 24], [0.15, 30, 22, 74], [0.32, 84, 30, 110], [0.50, 150, 44, 106],
    [0.66, 212, 72, 76], [0.82, 246, 136, 40], [0.93, 250, 200, 80], [1.00, 252, 250, 190],
], dtype=np.float64)


def colormap(values: np.ndarray) -> np.ndarray:
    """values in 0..1 -> uint8 RGB."""
    v = np.clip(values, 0.0, 1.0)
    out = np.empty(v.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(v, _ANCHORS[:, 0], _ANCHORS[:, c + 1]).astype(np.uint8)
    return out


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (h, w, 3) uint8 array as PNG."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def spectrogram_png(db: np.ndarray, floor: float, top: float, scale: int = 3) -> bytes:
    """dB spectrogram (frames, bands) -> PNG with time left to right, low frequencies at the bottom."""
    norm = (np.asarray(db, dtype=np.float64) - floor) / max(top - floor, 1e-9)
    img = colormap(norm.T[::-1])  # bands become rows, highest band on top
    img = np.repeat(np.repeat(img, scale, axis=0), 2, axis=1)
    return png_bytes(np.ascontiguousarray(img))
