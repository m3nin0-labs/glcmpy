#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Frontend for `xarray.DataArray`."""

from __future__ import annotations

import numpy as np
import xarray as xr

from glcmpy.handler import Angles, glcm_dispatch


@glcm_dispatch.register(xr.DataArray)
def _glcm_dataarray(
	data: xr.DataArray,
	measure: str,
	*,
	window_size: int,
	angles: Angles,
	n_grey: int,
	value_range: tuple[float, float] | None,
	rescale: bool,
) -> xr.DataArray:
	"""Compute a GLCM measure on an `xarray.DataArray`.

	The result is a new `xarray.DataArray` sharing the input dims, coordinates
	and attributes, with the measure parameters added under ``glcm_*`` keys. The
	input is left unchanged.

	Returns:
		xarray.DataArray: The new `xarray.DataArray` of the same shape as ``data``.

	Raises:
		ValueError: If the input is not a `xarray.DataArray`.

	See:
		See `glcmpy.handler.Measure` for the arguments.
	"""
	# compute the GLCM measure
	result = glcm_dispatch(
		data.values,
		measure,
		window_size=window_size,
		angles=angles,
		n_grey=n_grey,
		value_range=value_range,
		rescale=rescale,
	)

	# create the new `xarray.DataArray`
	out = data.copy(data=result)

	# set the attributes
	out.attrs = {
		**data.attrs,
		"glcm_measure": measure,
		"glcm_window_size": int(window_size),
		"glcm_n_grey": int(n_grey),
		"glcm_angles": np.atleast_1d(np.asarray(angles, dtype=np.float64))
		.ravel()
		.tolist(),
	}

	# return the new `xarray.DataArray`
	return out
