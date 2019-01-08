import os
import sys
from sdepy import __version__ as version

needs_sphinx = '1.1'


# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    ]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'SdePy'
copyright = '2018-2019, MC'
author = 'MC'
version = version
release = version
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
# import glob
# autosummary_generate = [master_doc + source_suffix] #glob.glob("*.txt")


# -----------------------------------------------------------------------------
# Options for HTML output
# -----------------------------------------------------------------------------

# html_theme = 'alabaster'
html_theme = 'classic'

html_theme_options = {}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'sdepydoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': '',  # latex_preamble,

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_toplevel_sectioning = 'part'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'sdepy_manual.tex',
     'SdePy Package Documentation',
     'MC', 'manual'),
]
