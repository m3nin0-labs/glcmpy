#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""glcmpy - GLCM (Grey-Level Co-occurrence Matrix) texture measures."""

from __future__ import annotations

from glcmpy.frontends import load_frontends
from glcmpy.handler import (
	asm,
	contrast,
	correlation,
	dissimilarity,
	energy,
	homogeneity,
	mean,
	std,
	variance,
)
from glcmpy.rescale import quantize, rescale

# register optional frontends
load_frontends()

#
# Version
#
__version__ = "0.1.0"

#
# Public API
#
__all__ = (
	"contrast",
	"dissimilarity",
	"homogeneity",
	"energy",
	"asm",
	"mean",
	"variance",
	"std",
	"correlation",
	"rescale",
	"quantize",
	"__version__",
)
