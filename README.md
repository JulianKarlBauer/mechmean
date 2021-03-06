[![PyPI version](https://badge.fury.io/py/mechmean.svg)][url_pypi_this_package]
[![Documentation status](https://readthedocs.org/projects/mechmean/badge/?version=latest)][url_read_the_docs_latest]
[![DOI](https://zenodo.org/badge/403947937.svg)][url_latest_doi]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JulianKarlBauer/mechmean/HEAD)



<!-- <p align="center">
  <a href="https://github.com/JulianKarlBauer/?">
  <img alt="PlanarFibers" src="logo/logo.png" width="20%">
  </a>
</p> -->

# Mechmean

This Python package contains selected mean field methods
in the context of continuum mechanics
with special focus on orientation averaging homogenization
<!--
and is utilied , e.g., in
```bibtex
@article{insertdoihere?,
	author = {Julian Karl Bauer and Thomas Böhlke},
	title ={On the dependence of orientation averaged mean field homogenization on planar fourth order fiber orientation tensors},
	journal = {?},
}
``` -->

The implementation is oriented as close as possible to the cited references
and no emphasis is placed on run time optimization.
Therefore, this package should be considered as a reference implementation
which can be used to cross-validate performance-optimized implementation.

Please see [license][url_license],
[acknowledgment](#acknowledgment)
and cite the latest [Zenodo-DOI][url_latest_doi]
<!-- and the [paper given above][url_article]. -->

## Installation [![PyPI version](https://badge.fury.io/py/mechmean.svg)][url_pypi_this_package]

Install with `pip` following instructions on [Python Package Index][url_pypi_this_package], i.e.,

```bash
pip install mechmean
```

**or** install from local files

- [Clone][url_how_to_clone] this repository to your machine
- Open a terminal and navigate to your local clone
- Install the package from the local clone into the current [env][url_env_python]i[ronment][url_env_conda] in develop mode:
	```shell
	python setup.py develop
	```

Note: [Develop vs. install](https://stackoverflow.com/a/19048754/8935243)

## Examples

Both example notebooks and example scripts are rendered [here][url_read_the_docs_latest_notebooks] and given as source [here](docs/source/notebooks).

## Acknowledgment

The research documented in this repository has been funded by the 
[German Research Foundation (DFG, Deutsche Forschungsgemeinschaft)][dfg_website] - project number [255730231][dfg_project].
The support by the German Research Foundation within the International Research Training Group 
[“Integrated engineering of continuous-discontinuous long fiber reinforced polymer structures“ (GRK 2078)][grk_website]
is gratefully acknowledged.

[grk_website]: https://www.grk2078.kit.edu/
[dfg_website]: https://www.dfg.de/
[dfg_project]: https://gepris.dfg.de/gepris/projekt/255730231

[url_license]: LICENSE
[url_latest_doi]: https://zenodo.org/badge/latestdoi/403947937
[url_article]: ??
[url_how_to_clone]: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

[url_env_python]: https://docs.python.org/3/tutorial/venv.html
[url_env_conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

[url_read_the_docs_latest]: https://mechmean.readthedocs.io/en/latest/
[url_read_the_docs_latest_notebooks]: https://mechmean.readthedocs.io/en/latest/source/example_notebooks.html
[url_pypi_this_package]: https://pypi.org/project/mechmean/

<!-- https://jacobtomlinson.dev/posts/2020/versioning-and-formatting-your-python-code/ -->
