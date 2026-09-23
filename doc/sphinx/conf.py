# © 2026. Triad National Security, LLC. All rights reserved.
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do
# so.

project = "flecsolve"
copyright = "2026, Triad National Security, LLC. All rights reserved."
author = "Los Alamos National Laboratory"

version = "0.0.1"
release = "0.0.1"

extensions = [
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
}
master_doc = "index"
language = "en"

pygments_style = "sphinx"
nitpicky = True

html_theme = "furo"
html_static_path = ["_static"]
html_title = "flecsolve documentation"
html_logo = "_static/flecsolve.svg"
html_theme_options = {
    "sidebar_hide_name": True,
}
