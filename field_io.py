import os
import numpy as np


def read_field_table(file_path: str):
    """
    Read a field table from a text file.

    Expected columns:
        x, y, z, Bx, By, Bz

    Unit convention:
        x, y, z are converted from mm to m by dividing by 1000.

    Returns
    -------
    tuple of np.ndarray
        x, y, z, Bx, By, Bz
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Field table not found: {file_path}")

    data = np.loadtxt(file_path)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError("Field table must contain at least 6 columns: x y z Bx By Bz")

    return (
        data[:, 0] / 1000.0,
        data[:, 1] / 1000.0,
        data[:, 2] / 1000.0,
        data[:, 3],
        data[:, 4],
        data[:, 5],
    )
