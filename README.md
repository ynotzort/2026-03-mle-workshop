# 2026-03-mle-workshop

## day 1

this project is based on https://github.com/ynotzort/ml-engineering-contsructor-workshop

## how to install uv

just run `curl -LsSf https://astral.sh/uv/install.sh | sh`


## day 1 steps
- create a folder `day_1`
- change directory into `day_1` (`cd day_1`)
- get the original notebook:
```bash
mkdir notebooks
cd notebooks
wget "https://raw.githubusercontent.com/ynotzort/2025-07-mle-workshop/refs/heads/main/day_1/notebooks/duration-prediction-starter.ipynb"
cd ..
```
- initialize a uv project with `uv init --python 3.10`
- run `uv sync`
