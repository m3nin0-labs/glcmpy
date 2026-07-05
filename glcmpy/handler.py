#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""GLCM texture measures.

Every measure is a pure function: it takes a 2-D image (a
`numpy.ndarray` or an `xarray.DataArray`) and returns a new object of the same
type. The input is never modified in place, so calls compose naturally.

Input handling uses `functools.singledispatch`: `glcm_dispatch` holds the
`numpy.ndarray` implementation and rejects unsupported types; optional frontends
(e.g. xarray) register themselves against it at import time.

Angle convention (radians, from the `sits` package):

* ``0``      -> neighbour to the right
* ``pi/4``   -> top-right diagonal
* ``pi/2``   -> above (vertical)
* ``3*pi/4`` -> top-left diagonal

The co-occurrence matrix is symmetric (opposite directions are merged). When
several angles are given, the per-angle results are averaged.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Protocol

import numpy as np
import numpy.typing as npt

#
# Constants
#
Angles = float | Sequence[float]

#
# Measurements
#
MEASURES = (
	"contrast",
	"dissimilarity",
	"homogeneity",
	"energy",
	"asm",
	"mean",
	"variance",
	"std",
	"correlation",
)

MEASURE_DEFAULT_DOC = """Compute the GLCM **{measure}** texture measure.

Args:
	data (npt.ArrayLike): 2-D grey-level image (`numpy.ndarray` or `xarray.DataArray`).

	window_size (int): Odd sliding-window size. Defaults to 3.

	angles (Angles): Direction(s) in radians. Results are averaged over several
		angles. Defaults to ``(0.0,)``.

	n_grey (int): Number of grey levels used to build the co-occurrence matrix.
		Defaults to 1000.

	value_range (tuple[float, float] | None): Source ``(low, high)`` range used 
		when quantizing. Defaults to the data min/max.

	rescale (bool): If True, quantize `data` into `[0, n_grey)` first. Set False
		when the input already holds integer grey levels. Defaults to True.

Returns:
	npt.NDArray[np.float64]: A new array of the same type and shape as ``data``. 
	The input is left unchanged.

Raises:
	TypeError: If ``data`` is neither a `numpy.ndarray` nor a supported
		frontend type.

	ValueError: If ``data`` is not 2-D, ``window_size`` is not a positive odd
		integer, or ``angles`` does not hold between 1 and 4 values.
"""


#
# Internal functions
#
def _make(measure: str) -> Measure:
	"""Build a public measure function bound to `measure`.

	Args:
		measure (str): The name of the measure to build.

	Returns:
		Measure: A callable measure function.
	"""

	# build the measure function
	def measure_func(
		data: npt.ArrayLike,
		*,
		window_size: int = 3,
		angles: Angles = (0.0,),
		n_grey: int = 1000,
		value_range: tuple[float, float] | None = None,
		rescale: bool = True,
	) -> npt.NDArray[np.float64]:
		# call the dispatch function
		return glcm_dispatch(
			data,
			measure,
			window_size=window_size,
			angles=angles,
			n_grey=n_grey,
			value_range=value_range,
			rescale=rescale,
		)

	# set the name, qualname and docstring
	measure_func.__name__ = measure
	measure_func.__qualname__ = measure
	measure_func.__doc__ = MEASURE_DEFAULT_DOC.format(measure=measure)

	# return the measure function
	return measure_func


#
# Classes
#
class Measure(Protocol):
	"""Signature shared by public GLCM measure functions."""

	def __call__(
		self,
		data: npt.ArrayLike,
		*,
		window_size: int = 3,
		angles: Angles = (0.0,),
		n_grey: int = 1000,
		value_range: tuple[float, float] | None = None,
		rescale: bool = True,
	) -> npt.NDArray[np.float64]: ...


#
# Public functions
#
@singledispatch
def glcm_dispatch(
	data: npt.ArrayLike,
	measure: str,
	*,
	window_size: int,
	angles: Angles,
	n_grey: int,
	value_range: tuple[float, float] | None,
	rescale: bool,
) -> npt.NDArray[np.float64]:
	"""Type-dispatch entry point.

	Raises:
		TypeError: If `data` is not a supported frontend type.
	"""
	raise TypeError(
		f"glcmpy does not support input of type {type(data).__name__!r}; "
		"pass a numpy.ndarray or an xarray.DataArray"
	)


#
# Public functions
#
contrast = _make("contrast")
dissimilarity = _make("dissimilarity")
homogeneity = _make("homogeneity")
energy = _make("energy")
asm = _make("asm")
mean = _make("mean")
variance = _make("variance")
std = _make("std")
correlation = _make("correlation")
