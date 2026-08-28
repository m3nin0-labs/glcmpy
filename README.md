# glcmpy

GLCM (Grey-Level Co-occurrence Matrix - Haralick) texture measures in Python. Built with [Eigen](https://libeigen.gitlab.io/).

## Install

The package is managed with [uv](https://docs.astral.sh/uv/). A base install builds the C++ core and pulls numpy:

```bash
uv sync
```

The `xarray.DataArray` frontend is optional. Enable it with the `xarray` extra:

```bash
uv sync --extra xarray
```

## Your first texture measure (numpy)

Start from a 2-D image. Here is a smooth gradient with a little noise, so texture varies across space.

```python
import numpy as np
import glcmpy

# random generation
rng = np.random.default_rng(0)

# generate image
image = (np.add.outer(np.arange(128), np.arange(128)) / 2.0) + rng.normal(0, 5, (128, 128))
```

Pick an odd `window_size` (the neighbourhood scanned around each pixel) and call a measure. The result is a new array of the same shape. The input is not changed.

```python
contrast = glcmpy.contrast(image, window_size=5)
```

Directions are given in radians (`0` = right, `pi/4` = top-right, `pi/2` = up, `3*pi/4` = top-left). Passing several angles averages the result, which makes it rotation-robust:

```python
correlation = glcmpy.correlation(
	image,
	window_size=5,
	angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
)
```

## From an xarray DataArray

With the `xarray` extra installed, the same measures accept a `DataArray` directly and return a new `DataArray`. Dimensions, coordinates and attributes are preserved, the measure parameters are recorded under `glcm_*` keys, and the input is never modified:

```python
import xarray as xr

# define xarray data
raster = xr.DataArray(image, dims=("y", "x"), name="reflectance")

# calculate variance
texture = glcmpy.variance(raster, window_size=5)   # new DataArray; `raster` untouched
```

## Quantization

The core works on integer grey levels in `[0, n_grey)`. By default each measure quantizes the input first (`rescale=True`, `n_grey=1000`). If your data already holds integer grey levels, pass `rescale=False`. You can also pre-quantize explicitly:

```python
# quantize
levels = glcmpy.quantize(image, n_grey=64, value_range=(0.0, 1.0))

# apply contrast
contrast = glcmpy.contrast(levels, n_grey=64, rescale=False)
```

## Learn more

Runnable [jupytext](https://jupytext.readthedocs.io) scripts live in `examples/`
(install the extras with `uv sync --group examples`):

- `examples/textures-numpy.py`, maps three measures over a four-texture image and
  plots them.
- `examples/textures-xarray.py`, runs a measure on a labelled `DataArray` and
  shows that structure is preserved.

## Development

Uses [uv](https://docs.astral.sh/uv/) + `scikit-build-core` + nanobind. Eigen is fetched automatically at build time (no system install needed).

```bash
uv sync --group dev --extra xarray
uv run ruff check . && uv run ruff format --check .
uv run mypy glcmpy
uv run pytest
```

## Documentation

To build the `glcmpy` documentation, you can use the following command:

```bash
uv run python scripts/build-docs.py   # build the pdoc site into ./site
```

## Acknowledgments

We would like to thank the developers and contributors of the [`sits`](https://github.com/e-sensing/sits) R package for their work on `GLCM` methods. The `glcmpy` is a standalone port of the texture functions from the [`sits`](https://github.com/e-sensing/sits) R package.

## License

Code is licensed under the **GNU General Public License v2.0**. See the [LICENSE](LICENSE) file. The texture math is ported from the [`sits`](https://github.com/e-sensing/sits) R package.
