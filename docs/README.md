## Build the Docs

### Setup (do once)

Install Doxygen:

```
brew install doxygen
```

Install Python packages:

```
pip install -r requirements.txt
```

Tiki uses this Sphinx/reStructuredText site, with Doxygen and Breathe for C++.
It does not use MkDocs. The Tiki layout pages require a Tiki build that includes
the Rust indexing extension, not only the upstream `mlx` package.

### Build

Build the docs from `mlx/docs/`

```
doxygen && make html
```

See `src/dev/tiki_layouts.rst` for the native extension build and executable
layout documentation checks.

View the docs by running a server in `mlx/docs/build/html/`:

```
python -m http.server <port>
```

and point your browser to `http://localhost:<port>`.

### Push to GitHub Pages

Check-out the `gh-pages` branch (`git switch gh-pages`) and build
the docs. Then force add the `build/html` directory:

`git add -f build/html`

Commit and push the changes to the `gh-pages` branch.

## Doc Development Setup

To enable live refresh of docs while writing:

Install sphinx autobuild
```
pip install sphinx-autobuild
```

Run auto build on docs/src folder
```
sphinx-autobuild ./src ./build/html
```
