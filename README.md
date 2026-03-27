# 2026-03-mle-workshop

# day 1

this project is based on https://github.com/ynotzort/ml-engineering-contsructor-workshop

## how to install uv

just run `curl -LsSf https://astral.sh/uv/install.sh | sh`


## day 1 steps

### get the notebook
- create a folder `day_1`
- change directory into `day_1` (`cd day_1`)
- get the original notebook:
```bash
mkdir notebooks
cd notebooks
wget "https://raw.githubusercontent.com/ynotzort/2025-07-mle-workshop/refs/heads/main/day_1/notebooks/duration-prediction-starter.ipynb"
cd ..
```

### create a uv project
- initialize a uv project with `uv init --python 3.10`
- run `uv sync`

### install dependencies
- `uv add scikit-learn==1.2.2 pandas pyarrow`
- `uv add --dev jupyter seaborn`
- now lets fix the error with numpy: `uv add numpy==1.26.4`

### launch jupyter notebook
- `uv run jupyter notebook`

### make vscode recognize the python env correctly and use jupyter from within
- open a .py file (main.py)
- on the bottom right click on the python version -> browse -> find the path to python (here it was /workspaces/2026-03-mle-workshop/day_1/.venv/bin/python)
- go to the jupyter notebook file and click select kernel -> python environments -> day_1

### Add Gitignore
- get gitignore from https://github.com/github/gitignore/blob/main/Python.gitignore and save it into .gitignore

### convert the notebook into a script
- `uv run jupyter nbconvert --to=script notebooks/duration-prediction-starter.ipynb`
- create a folder named `duration_prediction` (`mkdir duration_prediction`)
- move the freshly created file `notebooks/duration-prediction-starter.py` into `duration_prediction` and rename it to `train.py`

### lets make the train.py script nice
- remove the # lines from the script
- move all imports to the top
- remove matplotlib and seaborn
- try to run it: `uv run python duration_prediction/train.py`
- create a train function and remove top-level statements. add `if __name__=="__main__": ...`
- add the pipeline code
- parametrize the train function
- use argparse for argument parsing
    - alternatives are https://github.com/fastapi/typer and click and fire
- add docstrings and typing
- add simple error handling
- add logging: `uv add loguru`
- split out the argparse into main.py and make it a module by adding a `__init__.py` file. now we have tocall our code using `uv run python -m duration_prediction.main --train-date 2022-01 --val-date 2022-02 --model-save-path model.bin`

### create a make file
now we can run training via `make train`

### tests
- `uv add pytest`
- `mkdir tests`
- create a `__init__.py` file in the `tests` folder (`touch tests/__init__.py`)
- create a `test_train.py` file in the tests folder (has to start with `test_`)
- run tests via `uv run pytest` or `make test`


# day 2

## create the project and add the dependencies
- create a top level folder `day_2` just below `day_1` and change into it (`mkdir day_2 && cd day_2`)
- create a new uv project: `uv init --lib --python 3.10 duration_pred_serve`
- change dir into `duration_pred_serve` (via `cd duration_pred_serve`)
- add dependencies from day 1: `uv add scikit-learn==1.2.2 numpy==1.26.4`
- lets add testing and logging dependencies `uv add pytest loguru`
- add webserver dependency: `uv add "fastapi[standard]"`
- add requests dependency `uv add --dev requests`
- copy model over from day_1 (`mkdir models && cp ../../day_1/models/2022-01.bin models/`)

## ping example for fastAPI
- create a `ping.py` file inside of `src/duration_pred_serve/` and open it.
- change the python virtual environment to use the correct day 2 env. click on the bottom left where it says day_1 and click browse and select `/workspaces/2026-03-mle-workshop/day_2/duration_pred_serve/.venv/bin/`
- run it via `uv run fastapi dev src/duration_pred_serve/ping.py`

## implement serve
- implement simple loading of the model file, and run it via `uv run python src/duration_pred_serve/serve.py`