#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Value rescaling and quantization helpers.

The GLCM core works on integer grey levels in `[0, n_grey)`. Real imagery
(e.g., reflectance, spectral indices, etc) is continuous, so it must first be rescaled
into that range and quantized. These helpers mirror the normalization performed
by the `sits` R package (`texture_normalize` in `api_texture.R`).

All functions are pure: they return new arrays and never mutate their input.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def rescale(
	data: npt.ArrayLike,
	src_range: tuple[float, float],
	dst_range: tuple[float, float],
) -> npt.NDArray[np.float64]:
	"""Linearly rescale ``data`` from ``src_range`` to ``dst_range``.

	Applies ``(x - src_lo) / (src_hi - src_lo) * (dst_hi - dst_lo) + dst_lo``.

	Args:
		data (npt.ArrayLike): The input data to rescale.

		src_range (tuple[float, float]): The source `(low, high)` range of the data.

		dst_range (tuple[float, float]): The destination `(low, high)` range.

	Returns:
		npt.NDArray[np.float64]: A new `float64` array rescaled to the `dst_range`.

	Raises:
		ValueError: If `src_range` has no span (equal bounds).
	"""
	# get source range
	src_lo, src_hi = src_range

	# get destination range
	dst_lo, dst_hi = dst_range

	# check if the source range has a span
	span = src_hi - src_lo

	# if no span, raise an error
	if span == 0:
		raise ValueError("src_range must have distinct lower and upper bounds")

	# normalize the data
	normalized = (np.asarray(data, dtype=np.float64) - src_lo) / span

	# rescale the data
	return normalized * (dst_hi - dst_lo) + dst_lo


def quantize(
	data: npt.ArrayLike,
	n_grey: int,
	value_range: tuple[float, float] | None = None,
) -> npt.NDArray[np.float64]:
	"""Quantize continuous ``data`` to integer grey levels in ``[0, n_grey)``.

	Mirrors the `sits` texture pipeline: NaNs are filled with 0, the source
	range is taken from `value_range` (or the finite data min/max), values are
	clamped to it, then linearly rescaled to `[0, n_grey - 1]` and truncated.

	Args:
		data (npt.ArrayLike): The input data to quantize.

		n_grey (int): The number of grey levels to quantize to.

		value_range (tuple[float, float] | None): The `(low, high)` range
			to quantize from. Defaults to the finite min/max of `data`.

	Returns:
		npt.NDArray[np.float64]: A new `float64` array of integer-valued grey
		levels, ready for the core (which expects a float image and casts internally).

	Raises:
		ValueError: If `n_grey` is less than 2.
	"""
	# check if the number of grey levels is valid
	if n_grey < 2:
		raise ValueError("n_grey must be >= 2")

	# convert the data to a float array
	image = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)

	# get the value range
	if value_range is None:
		# use the finite min and max
		lo, hi = float(np.min(image)), float(np.max(image))

	else:
		# use the provided value range
		lo, hi = value_range

		# clip the data to the value range
		image = np.clip(image, lo, hi)

	# if the value range is constant, return a zero array
	if hi == lo:
		return np.zeros_like(image)

	# rescale the data to the integer range
	return np.trunc(rescale(image, (lo, hi), (0.0, n_grey - 1)))
