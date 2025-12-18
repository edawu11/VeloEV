# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('tutorials'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VeloEV'
copyright = '2025, Yida Wu'
author = 'Yida Wu'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      
    'sphinx.ext.napoleon',     
    'sphinx.ext.viewcode',    
    'nbsphinx',                
    'sphinx.ext.mathjax',      
]

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '.ipynb_checkpoints',
    '**/.ipynb_checkpoints',
    '**/.ipynb_checkpoints/**',
    '*-checkpoint.rst',
    '**/*-checkpoint.rst',
    '.*',
    '**/.*',
]
nbsphinx_allow_errors = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
nbsphinx_execute = 'never'
