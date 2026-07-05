#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# glcmpy is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Build the glcmpy documentation site with pdoc."""

from __future__ import annotations

from pathlib import Path

import pdoc

import glcmpy

#
# Constants
#
ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "site"


#
# Public functions
#
def build() -> None:
	"""Render the glcmpy welcome page and API reference into `OUTPUT_DIR`.

	The README is injected as the package docstring so the single ``glcmpy``
	page carries the project overview followed by every public member.
	"""
	glcmpy.__doc__ = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

	# configure the pdoc renderer
	pdoc.render.configure(docformat="google")

	# render the documentation
	pdoc.pdoc("glcmpy", output_directory=OUTPUT_DIR)


if __name__ == "__main__":
	build()
	print(f"docs written to {OUTPUT_DIR}")
