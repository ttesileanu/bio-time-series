# bio-time-series
Biological implementations of time series segmentation algorithms

## Installation

This package uses Python 3.8 or higher.

After downloading / cloning the repository, `chmod` to the folder containing `setup.py`,
and run

    pip install .
    
It is recommended to do this in a virtual environment. Before installing this pacakge,
install prerequisite packages in as listed in `setup.py`, under `install_requires`.
    
If the goal is to make changes to the code, an editable install is useful because it
always gives access to the newest version. This can be achieved using

    pip install -e .
    
## Examples

A number of usage examples are present in the `examples/test` folder. It's best to think
of these as read-only examples that can be viewed directly online, on GitHub. To run
interactively or tinker with the examples, it's best to make a copy first.

A good starting point is `example_integration_bio_wta`, as it uses a good fraction of
the code. 