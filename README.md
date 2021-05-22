# Welcome to mgng

A scientific python package. This project has been created with [the bs-science cookiecutter](git clone git@bitbucket.org:StefanUlbrich/bs-science-cookiecutter.git).

## Installation and development

First make sure to install Python (^3.7) the dependency management
tool [Poetry](https://python-poetry.org/) then create an isolated virtual
environment and install the dependencies.

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

