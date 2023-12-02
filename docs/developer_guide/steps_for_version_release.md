
# Version release guide  

---
## Next steps describe standard proces for version release utilising:  

## setuptools+pip+twine  

* Ensure you finished all code updates/changes/bug fixes/features
* Update version number in [pyproject.toml](../../pyproject.toml)
* Update dependencies in [pyproject.toml](../../pyproject.toml)
* Update documentation
* Commit+Pull+Push all changes
* Merge into Master/Main branch and Tag version number
* Build package:
    * `python -m build`
* Upload package to the PyPi/Pypi_test/Azure artifacts, etc...
    * Using twine: `python -m twine upload --repository testpypi dist/*`

Useful links:  
[setuptools quickstart](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)

---
**P.S**  
Consider to try these package managing tools:  
[hatch](https://hatch.pypa.io/latest/version/)  
[flit](https://flit.pypa.io/en/latest/)  
[pdm](https://pdm-project.org/latest/)  
[poetry](https://python-poetry.org/)  
