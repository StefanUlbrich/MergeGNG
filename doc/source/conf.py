# -*- coding: utf-8 -*-
# pylint: disable=R,C,W

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2021, Stefan Ulbrich, All rights reserved."
__license__ = "All rights reserved"
__maintainer__ = "Stefan Ulbrich"
__date__ = "22/05/2021"

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "mgng"
copyright = "2022, Stefan Ulbrich"
author = "Stefan Ulbrich"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


# Specify the baseurls for the projects I want to link to

# To find how to correctly link (if it does not work out of the box:)
# 1. pipenv run python -msphinx.ext.intersphinx https://scikit-learn.org/stable/objects.inv | less
#    pipenv run python -msphinx.ext.intersphinx https://docs.scipy.org/doc/scipy/reference/objects.inv | less
# 2. Look for the symbol
# 3. Scroll up until you find :py:data:, :py:class: or similar (the first entry that is not indented)
# 4. Use this for the reference

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": (" https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}  # List of patterns, relative to source directory, that match files and


# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [".ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_book_theme" # Currently, does not support sphinx5
html_theme = "pydata_sphinx_theme"

# html_favicon = "_static/favicon.ico"
# html_logo = "_static/bs.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Don't sort members alphabetically
# Improves readability when using the attrs package
# https://stackoverflow.com/questions/37209921/python-how-not-to-sort-sphinx-output-in-alphabetical-order
autodoc_member_order = "bysource"
