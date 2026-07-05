#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the optional xarray frontend."""

import numpy as np
import pytest

import glcmpy

xr = pytest.importorskip("xarray")


def _make_dataarray():
	"""Build a small labelled DataArray with coords and attrs."""
	# generate a random image
	data = np.random.default_rng(0).random((8, 10))

	# create the DataArray
	return xr.DataArray(
		data,
		dims=("y", "x"),
		coords={"y": np.arange(8), "x": np.arange(10) * 2.0},
		attrs={"long_name": "ndvi", "units": "1"},
		name="ndvi",
	)


def test_returns_new_dataarray_preserving_structure() -> None:
	"""Test that the output is a new DataArray with the same shape and coords."""
	# arrange
	da = _make_dataarray()

	# act
	out = glcmpy.variance(da, window_size=3)

	# assert
	assert isinstance(out, xr.DataArray)
	assert out is not da
	assert out.dims == da.dims
	assert out.shape == da.shape

	# check that the coords are the same
	for key in da.coords:
		np.testing.assert_array_equal(out.coords[key].values, da.coords[key].values)


def test_input_dataarray_unchanged() -> None:
	"""Test that the input DataArray is unchanged."""
	# arrange
	da = _make_dataarray()
	snapshot = da.values.copy()

	# act
	glcmpy.homogeneity(da, window_size=5)

	# assert
	np.testing.assert_array_equal(da.values, snapshot)


def test_metadata_attrs_added() -> None:
	"""Test that the metadata attrs are added."""
	# arrange / act
	da = _make_dataarray()
	out = glcmpy.contrast(da, window_size=3, angles=(0.0,), n_grey=64)

	# assert
	assert out.attrs["glcm_measure"] == "contrast"
	assert out.attrs["glcm_window_size"] == 3
	assert out.attrs["glcm_n_grey"] == 64
	assert out.attrs["long_name"] == "ndvi"  # original attrs preserved


def test_matches_ndarray_path() -> None:
	"""Test that the output matches the numpy path."""
	# arrange
	da = _make_dataarray()

	# act
	out_da = glcmpy.contrast(da, window_size=3, n_grey=64)
	out_np = glcmpy.contrast(da.values, window_size=3, n_grey=64)

	# assert
	np.testing.assert_allclose(out_da.values, out_np)
