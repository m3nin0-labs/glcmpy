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
# # GLCM texture measures on an xarray DataArray
#
# When xarray is installed, every `glcmpy` measure also accepts an
# `xarray.DataArray` and returns a new `DataArray`: dims, coordinates and
# attributes are preserved, the measure parameters are recorded under `glcm_*`
# keys, and the input is never modified. This makes texture a drop-in step in a
# labelled raster pipeline.
#
# Run it with `uv run python examples/textures-xarray.py`, or open it as a
# notebook (Jupyter and VS Code read the `# %%` cells directly).

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import glcmpy

# %% [markdown]
# ## 1) Wrap a raster as a labelled DataArray
#
# We build a smooth ramp plus noise and attach real-world `y`/`x` coordinates
# and metadata, as you would get from a geospatial raster.

# %%
# define data generator
rng = np.random.default_rng(0)

# raster shape
rows, cols = 128, 160

# raster values
values = (
     (np.add.outer(np.arange(rows), np.arange(cols)) / 2.0) + 
     (rng.normal(0, 6, size=(rows, cols)))
)

# create data array
raster = xr.DataArray(
	values,
	dims = ("y", "x"),
	coords = {"y": np.arange(rows) * 10.0, "x": np.arange(cols) * 10.0},
	attrs = {"long_name": "surface_reflectance", "units": "1"},
	name = "reflectance",
)

raster

# %% [markdown]
# ## 2) Compute a texture measure
#
# The call signature is identical to the numpy path. Here we average `contrast`
# over four directions for a rotation-robust texture map.

# %%
# calculate contrast
texture = glcmpy.contrast(
	raster,
	window_size=7,
	n_grey=128,
	angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
)

texture

# %% [markdown]
# ## 3) Structure is preserved and the input is untouched
#
# The result carries the same dims and coordinates, the original attributes are
# kept, and the measure parameters are recorded alongside them.

# %%
print("same dims:      ", texture.dims == raster.dims)
print("same coords:    ", np.array_equal(texture["x"].values, raster["x"].values))
print("input untouched:", np.array_equal(raster.values, values))
print("recorded metadata:")

# show metadata
for key in ("glcm_measure", "glcm_window_size", "glcm_n_grey", "glcm_angles"):
	print(f"  {key} = {texture.attrs[key]}")


# %% [markdown]
# ## 4) Plot with xarray
#
# Because the result is a `DataArray`, `.plot()` labels the axes with the
# coordinates for free.

# %%
# define figure
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

# plot input raster
raster.plot(ax=left, cmap="gray")
left.set_title("input reflectance")

# plot texture
texture.plot(ax=right, cmap="magma")
right.set_title("GLCM contrast")

# tight and show
fig.tight_layout()
plt.show()
