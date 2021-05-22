# -*- coding: utf-8 -*-
# pylint: disable=R,C,W

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2021, Stefan Ulbrich, All rights reserved."
__license__ = "All rights reserved"
__maintainer__ = "Stefan Ulbrich"
__email__ = "stefan.frank.ulbrich@gmail.com"
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

project = 'mgng'
copyright = "2021, Stefan Ulbrich"
author = 'Stefan Ulbrich'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'nbsphinx'
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# Specify the baseurls for the projects I want to link to

# To find how to correctly link (if it does not work out of the box:)
# 1. pipenv run python -msphinx.ext.intersphinx https://scikit-learn.org/stable/objects.inv | less
#    pipenv run python -msphinx.ext.intersphinx https://docs.scipy.org/doc/scipy/reference/objects.inv | less
# 2. Look for the symbol
# 3. Scroll up until you find :py:data:, :py:class: or similar (the first entry that is not indented)
# 4. Use this for the reference

intersphinx_mapping = {
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': (' https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('http://scikit-learn.org/stable', None)
}# List of patterns, relative to source directory, that match files and

# Add markdown support
source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}


source_suffix = ['.rst', '.md', ]
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.ipynb_checkpoints']

# -- Options for notebooks ---------------------------------------------------

# Notebooks should never be executed.
# The notebooks in the source should have all output stored withing
# It would be prefereable to export to rst / md but this ruins the pandas tables

nbsphinx_execute = 'never'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "MergeGNG",
    # Set you GA account ID to enable tracking
    # "google_analytics_account": "UA-XXXXX",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # "base_url": "https://project.github.io/project",
    # Set the color and the accent color
    "color_primary": "blue",
    "color_accent": "light-blue",
    # Set the repo location to get a badge with stats
    # "repo_url": "https://github.com/project/project/",
    # "repo_name": "Project",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": -1,
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": True,
    "version_dropdown": False
}
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}


# html_theme = 'sphinx_rtd_theme'
html_favicon = "_static/favicon.ico"

html_logo ="_static/bs.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Don't sort members alphabetically
# Improves readability when using the attrs package
# https://stackoverflow.com/questions/37209921/python-how-not-to-sort-sphinx-output-in-alphabetical-order
autodoc_member_order = 'bysource'