[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/403947937.svg)](https://zenodo.org/badge/latestdoi/403947937)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JulianKarlBauer/?/HEAD)

<!-- <p align="center">
  <a href="https://github.com/JulianKarlBauer/?">
  <img alt="PlanarFibers" src="logo/logo.png" width="20%">
  </a>
</p> -->

# Mechmean

This Python package contains selected mean field methods
in the context of continuum mechanics
with special focus on orientation averaged homogenization
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
<!-- and the [paper given above][url_mms_article]. -->

## Installation

- [Clone][url_how_to_clone] this repository to your machine
- Open a terminal and navigate to your local clone
- Install the package from the local clone into the current [env][url_env_python]i[ronment][url_env_conda] in develop mode:
	```shell
	python setup.py develop
	```

Note: [Develop vs. install](https://stackoverflow.com/a/19048754/8935243)

## Examples

Both example notebooks and example scripts are given [here](examples/).

## Acknowledgment

The research documented in this repository has been funded by the German Research Foundation (DFG) within the
International Research Training Group [“Integrated engineering of continuous-discontinuous long fiber reinforced polymer structures“ (GRK 2078)][grk_website].
The support by the [German Research Foundation (DFG)][dfg_website] is gratefully acknowledged.

[grk_website]: https://www.grk2078.kit.edu/
[dfg_website]: https://www.dfg.de/

[url_license]: LICENSE
[url_latest_doi]: ??
[url_mms_article]: ??
[url_how_to_clone]: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

[url_env_python]: https://docs.python.org/3/tutorial/venv.html
[url_env_conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html



