#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Frontends for optional input types."""

import importlib

#
# Constants
#
FRONTENDS = (
	"glcmpy.frontends.np", 
	"glcmpy.frontends.xr",
)


#
# Public functions
#
def load_frontends() -> None:
	"""Import available frontend modules."""
	for module in FRONTENDS:
		try:
			importlib.import_module(module)
		except ImportError:
			continue
