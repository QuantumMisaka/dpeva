"""Lossless NPZ output, readable with standard numpy.load (no extra codec)."""
from zipfile import ZIP_LZMA, ZipFile

import numpy as np


def save_npz(path, **arrays):
    """Use ZIP LZMA instead of DEFLATE; preserve every array byte and dtype."""
    with ZipFile(path, 'w', compression=ZIP_LZMA, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
