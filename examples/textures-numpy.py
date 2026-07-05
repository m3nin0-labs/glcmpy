# %%
#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

# %% [markdown]
# # GLCM texture measures on a numpy image
#
# GLCM (Grey-Level Co-occurrence Matrix) texture measures summarise the *local*
# spatial arrangement of grey levels around every pixel. `glcmpy` slides a window
# over the image and, for each pixel, builds a co-occurrence matrix of
# neighbouring grey-level pairs and reduces it to a single number (contrast,
# homogeneity, correlation, ...).
#
# This walkthrough builds a synthetic image with four visibly different textures
# and maps each measure over it.
#
# Run it with `uv run python examples/textures-numpy.py`, or open it as a
# notebook (Jupyter and VS Code read the `# %%` cells directly).

# %%
import matplotlib.pyplot as plt
import numpy as np

import glcmpy

# %% [markdown]
# ## 1) Build an image with four distinct textures
#
# Each quadrant has its own spatial structure: a nearly flat patch, fine random
# noise, vertical stripes, and a smooth horizontal gradient. These are exactly
# the kinds of differences GLCM measures are designed to capture.


# %%
def texture_image(size: int = 192, seed: int = 0) -> np.ndarray:
	"""Build a square image with four different-texture quadrants.

	Args:
		size: Side length of the (square) output image in pixels.

		seed: Seed for the random noise quadrants.

	Returns:
		A ``(size, size)`` float image in roughly ``[0, 1]``.
	"""
	rng = np.random.default_rng(seed)
	image = np.empty((size, size), dtype=np.float64)
	half = size // 2

	# top-left: almost flat (very low texture)
	image[:half, :half] = 0.5 + rng.normal(0, 0.01, (half, half))

	# top-right: fine high-frequency noise (high contrast)
	image[:half, half:] = rng.random((half, half))

	# bottom-left: vertical stripes (a strongly directional texture)
	stripes = (np.arange(half) % 8 < 4).astype(np.float64)
	image[half:, :half] = np.tile(stripes, (half, 1))

	# bottom-right: smooth horizontal gradient
	image[half:, half:] = np.tile(np.linspace(0, 1, half), (half, 1))

	return image


# generate image
image = texture_image()

# show dimensions
print("image:", image.shape, image.dtype)

# %% [markdown]
# ## 2) Pick a window and quantization
#
# `window_size` is the odd side length of the neighbourhood scanned around each
# pixel; larger windows capture coarser texture. By default each measure
# quantizes the continuous image into `n_grey` integer grey levels before
# building the co-occurrence matrix (`rescale=True`).

# %%
WINDOW = 7
N_GREY = 64

# %% [markdown]
# ## 3) Map three measures over the image
#
# `contrast` grows with local intensity variation, `homogeneity` is high where
# neighbours are similar, and `correlation` measures linear grey-level
# dependence. Averaging over four directions makes the result rotation-robust.

# %%
# calculate contrast
contrast = glcmpy.contrast(image, window_size = WINDOW, n_grey = N_GREY)

# calculate homogeneity
homogeneity = glcmpy.homogeneity(image, window_size = WINDOW, n_grey = N_GREY)

# calculate correlation
correlation = glcmpy.correlation(
	image,
	window_size = WINDOW,
	n_grey = N_GREY,
	angles = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
)

# show information
for name, texture in ("contrast", contrast), ("homogeneity", homogeneity), ("correlation", correlation):
	print(f"{name:12s} min={texture.min():.4f} max={texture.max():.4f}")

# %% [markdown]
# ## 4) Visualise the input and its texture maps
#
# Each texture map has the same shape as the input, so quadrant boundaries line
# up. The noisy quadrant lights up under `contrast`; the flat and gradient
# quadrants stay bright under `homogeneity`.

# %%
# define panels
panels = (
	("input image", image, "gray"),
	("contrast", contrast, "magma"),
	("homogeneity", homogeneity, "magma"),
	("correlation", correlation, "magma"),
)

# prepare figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# iterate panels
for axis, (title, data, cmap) in zip(axes, panels, strict=True):
    # show data
	image_plot = axis.imshow(data, cmap=cmap)

    # configure title and other details
	axis.set_title(title)
	axis.axis("off")
	fig.colorbar(image_plot, ax=axis, fraction=0.046, pad=0.04)

fig.tight_layout()
plt.show()
