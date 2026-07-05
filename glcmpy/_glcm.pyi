#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Type stubs for the compiled ``glcmpy._glcm`` extension module."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

#
# Types
#
ImageArray = npt.NDArray[np.float64]

#
# Functions
#
def contrast(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def dissimilarity(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def homogeneity(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def energy(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def asm(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def mean(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def variance(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def std(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
def correlation(
	image: ImageArray, angles: Sequence[float], n_grey: int, window_size: int
) -> ImageArray: ...
