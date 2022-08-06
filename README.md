# Welcome to mgng

An experimental implementation of the Merge Growing Neural Gas algorithm.
The project turned into an exercise for using state-of-the-art python tools.
This project has been created with my [cookiecutter for science projects](https://bitbucket.org/StefanUlbrich/science-cookiecutter/src/master/)

## The Growing Neural Gas and Merge Growing Neural Gas Algorithms

[*Growing Neural Gas (NGN)*](https://papers.nips.cc/paper/893-a-growing-neural-gas-network-learns-topologies.pdf) is a [topology preserving (see this blog for a demonstration)](http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html) or [this explaination](http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html#id1)) extension to the [*Neural gas (NG)*]() approach is usefull for learning when an underlying topology is not known (as in the case of the [*Self-organizing maps (SOM)*]() algorithm). When it comes to time series data (such as trajectories), an extension to the neural gas algorithm has been approached (*Merge Neural Gas (MNG)*) and a combination with the GNG leads to the [*Merge growing neural gas (MNGN)*](https://ias.in.tum.de/_media/spezial/bib/andreakis09wsom.pdf) approach. It adds a context memory to the neurons of the NGN and is useful for *recognising* temporal sequences and with a single weighting parameter, can be reduced to a regular NGN for which an implementation [is available](https://github.com/itdxer/neupy/blob/master/notebooks/growing-neural-gas/Making%20Art%20with%20Growing%20Neural%20Gas.ipynb).

This packages implements the MGNG algorithm as a vanilla [numpy](https://numpy.org/) implementation (which can be executed on the GPU with [Cupy](https://cupy.chainer.org/)). The package uses modern python tools such as [poetry](https://python-poetry.org/), [attrs](https://www.attrs.org/en/stable/) (a focus has been laid on those two for this release), and sphinx and mypy/pylint/black for documentation and coding standards.

See the notebooks in the repective subfolder of the project root and the [documentation](https://stefanulbrich.github.io/MergeGNG/api/mgng.mgng.html).

## Installation and development

First make sure to install Python (^3.7) the dependency management
tool [Poetry](https://python-poetry.org/) then create an isolated virtual
environment and install the dependencies:

```sh
poetry install
```

Per terminal session,  the following command should be executed
to activate the virtual environment.

```sh
poetry shell
```

To generate the documentation run:

```sh
cd doc/
make api # optional, only when the code base changes
make html
```

To run unit tests, run:

```sh
pytest --log-level=WARNING
# Specify a selected test
pytest --log-level=DEBUG -k "TestExample"
pytest --log-level=DEBUG tests/test_example.py::TestExample::test_example
```

To work with [VisualStudio Code](https://code.visualstudio.com/):

```sh
cp .vscode/template.settings.json .vscode/settings.json
which python # copy the path without the executable
```

and add the path to the virtual environment to in the `"python.pythonPath"` setting.

```sh
cp .vscode/template.settings.json .vscode/settings.json
which python # copy the path without the executable
```

and add the path to the virtual environment to in the `"python.pythonPath"` setting.
