
# Documentation Guide

---
## MkDocs

**Packages in use for documentation purposes:**  

[mkdocs](https://www.mkdocs.org): documentation engine  
[mkdocstrings](https://mkdocstrings.github.io/): support docstrings in code  
[mkdocstrings-python](https://mkdocstrings.github.io/python/): python handler  
[mkdocs-gen-files](https://github.com/oprypin/mkdocs-gen-files): automatic api generation from docstrings  
[mkdocs-literate-nav](https://github.com/oprypin/mkdocs-literate-nav): automatic nav config creation for code API   
[mkdocs-section-index](https://github.com/oprypin/mkdocs-section-index): eliminate `__init__` headers in docs   
[mkdocs-material](https://github.com/squidfunk/mkdocs-material): custom theme

following great recipe:  [automatic-code-reference-pages](https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages)  
catalog of plugins and mods: [https://github.com/mkdocs/catalog](https://github.com/mkdocs/catalog)

### MkDocs Commands

* `mkdocs new [dir-name]` - Create a new project
* `mkdocs serve` - Start the live-reloading docs server
* `mkdocs build` - Build the documentation site
* `mkdocs gh-deploy --clean` - Upload documentation to [GitHub Pages](https://pages.github.com/)
* `mkdocs -h` - Print help message and exit

### Project layout
```
    mkdocs.yml    # The configuration file  
    docs/  
        index.md  # The documentation homepage  
        ...       # Other markdown pages, nested folders, images and other files  
```
### Adding new page (not API)

* Add xxx.md file to the `docs/`, for nested files create appropriate folders  
* Update mkdocs.yml nav config:  

  ```yaml
    nav:
      - Nested name 1:
        - Page name: 'nested_folder_1/xxx.md'
  ```

### API reference generation  
API generated automatically utilizing mkdocs-gen-files plugin  
Python script responsible for auto-generation: `scripts/gen_ref_pages.py`  

---
## Read The Docs
[Getting started with MkDocs](https://docs.readthedocs.io/en/stable/intro/getting-started-with-mkdocs.html)