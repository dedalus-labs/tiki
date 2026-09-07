# Copyright © 2023 Apple Inc.

# -*- coding: utf-8 -*-

import os
import subprocess

import tiki as tk

# -- Project information -----------------------------------------------------

project = "Tiki"
copyright = "2026 Dedalus Labs, Inc. Portions 2023 Apple Inc."
author = "Dedalus Labs"
version = ".".join(tk.__version__.split(".")[:3])
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "breathe",
]

python_use_unqualified_type_names = True
autodoc_type_aliases = {"Coordinate": "tiki.layout.composed.Coordinate"}
autosummary_generate = True
autosummary_filename_map = {
    "tiki.Stream": "stream_class",
    "tiki.PrintOptions": "printoptions_class",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

breathe_projects = {"tiki": "../build/xml"}
breathe_default_project = "tiki"

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["tiki.css"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
main_doc = "index"
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "attrs_block"]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]
highlight_language = "python"
pygments_style = "sphinx"
add_module_names = False

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Tiki"
html_baseurl = "https://oss.dedaluslabs.ai/tiki/"
html_show_sourcelink = False
html_favicon = "_static/favicon.png"

html_theme_options = {
    "logo": {
        "text": "Tiki",
        "image_light": "_static/tiki_mark.png",
        "image_dark": "_static/tiki_mark.png",
        "alt_text": "Tiki",
    },
    "github_url": "https://github.com/dedalus-labs/tiki",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_context = {
    "github_user": "dedalus-labs",
    "github_repo": "tiki",
    "github_version": "main",
    "doc_path": "docs/src",
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "tiki_doc"


def repository_link(app, env, node, contnode):
    """Included Markdown links to repository files; those resolve to the repository.

    The Tiki pages are the repository's own Markdown, included as they are, and
    they link to sources and sibling documents by relative path. A link that
    names a file in the checkout becomes a link to that file on GitHub.
    """
    from pathlib import Path

    from docutils import nodes

    target = node.get("reftarget", "")
    if node.get("refdomain") not in (None, "std") or "://" in target or target.startswith("#"):
        return None
    source = Path(env.doc2path(node.get("refdoc", env.docname)))
    root = Path(app.srcdir).parents[1]
    resolved = (source.parent / target.split("#")[0]).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None
    if not resolved.exists():
        return None
    context = app.config.html_context
    url = f"https://github.com/{context['github_user']}/{context['github_repo']}/blob/{context['github_version']}/{relative}"
    return nodes.reference("", "", contnode, internal=False, refuri=url)


def setup(app):
    from sphinx.util import inspect

    app.connect("missing-reference", repository_link)

    wrapped_isfunc = inspect.isfunction

    def isfunc(obj):
        type_name = str(type(obj))
        if "nanobind.nb_method" in type_name or "nanobind.nb_func" in type_name:
            return True
        return wrapped_isfunc(obj)

    inspect.isfunction = isfunc


# -- Options for LaTeX output ------------------------------------------------

latex_documents = [(main_doc, "Tiki.tex", "Tiki Documentation", author, "manual")]
latex_elements = {
    "preamble": r"""
    \usepackage{enumitem}
    \setlistdepth{5}
    \setlist[itemize,1]{label=$\bullet$}
    \setlist[itemize,2]{label=$\bullet$}
    \setlist[itemize,3]{label=$\bullet$}
    \setlist[itemize,4]{label=$\bullet$}
    \setlist[itemize,5]{label=$\bullet$}
    \renewlist{itemize}{itemize}{5}
""",
}
