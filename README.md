# aias_trading25

## Structure

`algos/` contains different optimisation alogorithms. They should be all subclassing `BaseAlgo`.

`datasets/` contains various datasets used for trading strategies. 

`signals.py` contains functions that generate new signal datasets. (they should be cached)

`bot.py` contains a sim_bot function.

Notebooks bring these together. `test_*.ipynb` are notebooks to test the functionality of the code. The rest are for the report. 

`optimise.ipynb` is the main notebook for the project.

## Getting started

You must first install the required packages. You can do this by running `pip install -r requirements.txt` (should create a virtualenv first).