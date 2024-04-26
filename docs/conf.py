from datetime import date
import os
import shutil

import toml

import audeer


config = toml.load(audeer.path("..", "pyproject.toml"))


# Project -----------------------------------------------------------------
author = ", ".join(author["name"] for author in config["project"]["authors"])
copyright = f"2020-{date.today().year} audEERING GmbH"
project = config["project"]["name"]
version = audeer.git_repo_version()
title = "Documentation"


# General -----------------------------------------------------------------
master_doc = "index"
source_suffix = ".rst"
exclude_patterns = [
    "api-src",
    "build",
    "tests",
    "Thumbs.db",
    ".DS_Store",
]
templates_path = ["_templates"]
pygments_style = None
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # support for Google-style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",  # for "copy to clipboard" buttons
    "sphinxcontrib.katex",  # has to be before jupyter_sphinx
    "matplotlib.sphinxext.plot_directive",  # include resulting figures in doc
]

# Do not copy prompot output
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Mapping to external documentation
intersphinx_mapping = {
    "audmath": ("https://audeering.github.io/audmath/", None),
    "audmetric": ("https://audeering.github.io/audmetric/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    "https://gitlab.audeering.com",
]

# Plotting
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_pre_code = ""
plot_rcparams = {
    "figure.figsize": "5, 3.8",  # inch
}
plot_formats = ["svg"]

# Disable auto-generation of TOC entries in the API
# https://github.com/sphinx-doc/sphinx/issues/6316
toc_object_entries = False


# HTML --------------------------------------------------------------------
html_theme = "sphinx_audeering_theme"
html_theme_options = {
    "display_version": True,
    "logo_only": False,
    "footer_links": False,
}
html_context = {
    "display_github": True,
}
html_title = title


# Copy API (sub-)module RST files to docs/api/ folder ---------------------
audeer.rmdir("api")
audeer.mkdir("api")
api_src_files = audeer.list_file_names("api-src")
api_dst_files = [
    audeer.path("api", os.path.basename(src_file)) for src_file in api_src_files
]
for src_file, dst_file in zip(api_src_files, api_dst_files):
    shutil.copyfile(src_file, dst_file)
