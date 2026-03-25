
# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "digiGAM"
author = "Domi Bruns"
copyright = "2025, Domi Bruns"

# Versioning
version = "2025.12"
release = "2025.12.1"

# -- Path setup --------------------------------------------------------------
# Make the project importable for autodoc (adjust as needed)
import os
import sys
# Assuming conf.py is at docs/source/conf.py, project root is two levels up
sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",               # Markdown support
    "sphinx.ext.autodoc",        # Pull docstrings
    "sphinx.ext.autosummary",    # Auto-generate API stubs
    "sphinx_autodoc_typehints",  # Render type hints nicely
    "sphinx.ext.napoleon",       # Uncomment if you use Google/NumPy-style docstrings
    # "sphinx.ext.intersphinx",  # Uncomment if you want cross-project links
    # "sphinx.ext.viewcode",     # Adds source code links
]
autosummary_generate = True

# Recognize both .rst and .md sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Internationalization
language = "en"

# Templates / patterns
templates_path = ["_templates"]
exclude_patterns = []

# MyST (Markdown) configuration
myst_enable_extensions = [
    "linkify",       # auto-link bare URLs
    "colon_fence",   # fenced directives like ```{toctree}
]
myst_heading_anchors = 3  # add anchors to h1–h3

# Autodoc defaults (optional, helpful)
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"  # move type hints into the description

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"   # keep one theme setting
html_static_path = ["_static"]

def setup(app):
    app.add_css_file('custom.css')

# -- HTML output -------------------------------------------------------------
# Start with a simple built-in theme; you can switch later
html_theme = "furo"

# Optional: set a nicer title or add a logo later
# html_title = f"{project} {release}"
# html_logo = "_static/logo.png"

# Optional: Intersphinx mapping if you enable "sphinx.ext.intersphinx"
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", {}),
#     "pandas": ("https://pandas.pydata.org/pandas-docs/stable", {}),
# }

def skip_undocumented_members(app, what, name, obj, skip, options):
    # Skip if the source has the magic skip comment
    try:
        source = getattr(obj, "__doc__", None)
        if source and "# sphinx-autodoc-skip" in source:
            return True
    except Exception:
        pass

    # Skip if assigned a special attribute manually
    if hasattr(obj, "__sphinx_autodoc_skip__"):
        return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_undocumented_members)
