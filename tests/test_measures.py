#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the numpy path of every GLCM measure."""

import math

import numpy as np
import pytest

import glcmpy

MEASURES = [
	"contrast",
	"dissimilarity",
	"homogeneity",
	"energy",
	"asm",
	"mean",
	"variance",
	"std",
	"correlation",
]

ANGLES = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]


@pytest.mark.parametrize("measure", MEASURES)
def test_shape_and_dtype(measure: str) -> None:
	"""Test that the output shape and dtype are correct."""
	# Arrange
	image = np.random.default_rng(0).random((12, 9))

	# Act
	out = getattr(glcmpy, measure)(image, window_size=3)

	# Assert
	assert out.shape == image.shape
	assert out.dtype == np.float64


@pytest.mark.parametrize("measure", MEASURES)
def test_input_unchanged(measure: str) -> None:
	"""Test that the input is unchanged."""
	# Arrange
	image = np.random.default_rng(1).random((10, 10))
	snapshot = image.copy()

	# Act
	getattr(glcmpy, measure)(image, window_size=5)

	# Assert
	np.testing.assert_array_equal(image, snapshot)


def test_uniform_image_analytic() -> None:
	"""Test that the output is correct for a uniform image."""
	# define a constant image quantizes to a single 
	# grey level -> p(0, 0) = 1
	image = np.full((8, 8), 3.14)

	# Act / Assert
	assert np.allclose(glcmpy.contrast(image, window_size=3), 0.0)
	assert np.allclose(glcmpy.homogeneity(image, window_size=3), 1.0)
	assert np.allclose(glcmpy.asm(image, window_size=3), 1.0)
	assert np.allclose(glcmpy.energy(image, window_size=3), 1.0)
	assert np.allclose(glcmpy.variance(image, window_size=3), 0.0)

	# near-zero variance: correlation is special-cased to 1
	assert np.allclose(glcmpy.correlation(image, window_size=3), 1.0)


@pytest.mark.parametrize("measure", MEASURES)
def test_multi_angle_is_mean_of_singles(measure: str) -> None:
	"""Test that the output is the mean of the single-angle results."""
	# Arrange
	image = np.random.default_rng(3).integers(0, 8, size=(10, 10)).astype(np.float64)
	measure_fn = getattr(glcmpy, measure)

	# Act
	singles = [
		measure_fn(
			image,
			window_size=3,
			angles=angle,
			n_grey=8,
			rescale=False,
		)
		for angle in ANGLES
	]
	combined = measure_fn(
		image,
		window_size=3,
		angles=ANGLES,
		n_grey=8,
		rescale=False,
	)

	# Assert
	np.testing.assert_allclose(
		combined, np.mean(singles, axis=0), atol=1e-12
	)


def test_invalid_window_size_raises() -> None:
	"""Test that an invalid window size raises a ValueError."""
	with pytest.raises(ValueError):
		glcmpy.contrast(np.zeros((5, 5)), window_size=4)


def test_non_2d_input_raises() -> None:
	"""Test that a non-2D input raises a ValueError."""
	with pytest.raises(ValueError):
		glcmpy.contrast(np.zeros((3, 3, 3)))


@pytest.mark.parametrize("bad", [[1, 2, 3], 5, "image", {"a": 1}, None])
def test_unsupported_type_raises_typeerror(bad: object) -> None:
	"""Test that an unsupported type raises a TypeError."""
	with pytest.raises(TypeError):
		glcmpy.contrast(bad)
