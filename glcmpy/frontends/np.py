#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Frontend for `numpy.ndarray`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from glcmpy import _glcm
from glcmpy.handler import Angles, glcm_dispatch
from glcmpy.rescale import quantize


#
# Internal functions
#
def _normalize_angles(angles: Angles) -> list[float]:
	"""Coerce `angles` to a list of 1 to 4 floats."""
	# coerce the angles to a 1-D array of floats
	values = np.atleast_1d(np.asarray(angles, dtype=np.float64)).ravel()

	# check if the number of angles is between 1 and 4
	if not 1 <= values.size <= 4:
		raise ValueError("angles must contain between 1 and 4 values")

	# return the angles as a list of floats
	return values.tolist()


def _check_window(window_size: int) -> int:
	"""Validate `window_size` as a positive odd integer."""
	# check if the window size is a positive odd integer
	if window_size < 1 or window_size % 2 == 0:
		raise ValueError("window_size must be a positive odd integer")

	# return the window size as an integer
	return int(window_size)


@glcm_dispatch.register
def _glcm_ndarray(
	data: np.ndarray,
	measure: str,
	*,
	window_size: int,
	angles: Angles,
	n_grey: int,
	value_range: tuple[float, float] | None,
	rescale: bool,
) -> npt.NDArray[np.float64]:
	"""GLCM measure implementation for `numpy.ndarray`."""
	# check if the data is 2-D
	if data.ndim != 2:
		raise ValueError(f"expected a 2-D array (got {data.ndim} dims)")

	# validate the window size
	window = _check_window(window_size)

	# normalize the angles
	resolved_angles = _normalize_angles(angles)

	# quantize the data if needed
	image = (
		quantize(data, n_grey, value_range)
		if rescale
		else np.ascontiguousarray(data, dtype=np.float64)
	)

	# get the kernel function
	kernel = getattr(_glcm, measure)

	# compute the GLCM measure
	return kernel(image, resolved_angles, int(n_grey), window)
