# ml-zoomcamp-playground
This project is intended to use for the practicing of the labs and submitting the homework for [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Modules
 - 01: Introduction to Machine Learning
 - 02: Machine Learning for Regression
 - 03: Machine Learning for Classification
 - 04: Evaluation Metrics for Classification
 - 05: Deploying Machine Learning Models

## User Guide
The following is the brief guide to set up the project.

#### Creating virtual env

```bash
uv venv .venv
```

#### Installing dependencies
 - uv
 - numpy
 - pandas
 - scikit-learn
 - seaborn
 - jupyter
 - requests

```bash
uv sync --locked
```

#### Creating and adding a separate ipython kernel for the project

```bash
uv add --dev ipykernel

uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=ml-zoomcamp-playground
```

#### Running jupyter notebook

```bash
uv run --with jupyter jupyter lab
```

#### To run Jupyter-notebook in Codespaces
```bash
jupyter notebook --no-browser --port 8888
```
