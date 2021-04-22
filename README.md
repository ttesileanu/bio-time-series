# bio-time-series

Biological implementations of time-series segmentation algorithms. This repository was developed for the paper *Neural circuits for dynamics-based segmentation of time series* by Tiberiu Tesileanu, Siavash Golkar, Samaneh Nasiri, Anirvan M. Sengupta, and Dmitri B. Chklovskii.

## Installation

This package uses Python 3.6 or higher.

After downloading or cloning the repository, ensure first that the prerequisite pacakges listed in `setup.py` under `install_requires` are installed. It is recommended to do so in a virtual environment.

Next, in a terminal ensure that the current directory is the one containing `setup.py` and run

    pip install .

If the goal is to make changes to the code, an editable install is useful because it always gives access to the newest version. This can be achieved using

    pip install -e .

## Reproducing results from the paper

The results from the paper can be reproduced using the script at [draft/neural_comp_draft.py](https://github.com/ttesileanu/bio-time-series/blob/master/draft/neural_comp_draft.py). This is a Jupyter notebook converted to a Python script using [Jupytext](https://github.com/mwouts/jupytext). You can run the script as-is or use [Jupytext](https://github.com/mwouts/jupytext) to convert the script back to a notebook before using.

## Examples

There are also a number of usage examples that double up as tests of the framework. They can be found in the [examples/test](https://github.com/ttesileanu/bio-time-series/tree/master/examples/tests) folder. A particularly good starting point is [example_integration_bio_wta.ipynb](https://github.com/ttesileanu/bio-time-series/blob/master/examples/tests/example_integration_bio_wta.ipynb), as it uses a good fraction of the code.

## Unit tests

Apart from the example notebooks that serve as integration tests, there are also a number of unit tests that you can run to ensure that the code is running properly on your machine. To run, `cd` to the `tests` folder and run

    python -m unittest discover .

If you get any errors, make sure you activated the environment that you used to install the package, and that all prerequisites are satisfied.

## Issues?

If you encounter any issues with the package, plese file a report at https://github.com/ttesileanu/bio-time-series/issues.
