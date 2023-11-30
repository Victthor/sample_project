
# Version release guide

## Next steps describe standard proces for version release utilising:  

## setuptools+pip+twine  

* Ensure you finished all code updates/changes/bug fixes/features
* Update version number in [pyproject.toml](../pyproject.toml)
* Update dependencies in [pyproject.toml](../pyproject.toml)
* Commit+Pull+Push all changes
* Merge into Master/Main branch and Tag version number
* Build package:
    * `python -m build`
* Upload package to the PyPi/Pypi_test/Azure artifacts, etc...
    * Using twine: `python -m twine upload --repository testpypi dist/*`

Useful links:  
https://setuptools.pypa.io/en/latest/userguide/quickstart.html

#### P.S
Consider to try these package managing tools:  
https://hatch.pypa.io/latest/version/  
https://flit.pypa.io/en/latest/  
https://pdm-project.org/latest/  
https://python-poetry.org/  
