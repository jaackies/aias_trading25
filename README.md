# aias_trading25

## Structure

`algos/` contains different optimisation alogorithms. They should be all subclassing `BaseAlgo`.

`datasets/` contains various datasets used for trading strategies. Each dataset should be a subclass of `BaseDataset`. 

`signals.py` contains functions that generate new signal datasets. (they should be cached)

`eval.py` contains a function that takes in a dataset and a series of parameters and returns cash held.

`optimise.py` takes in a dataset and an optimisation algorithm and uses the optimisation algorithm to optimise the parameters of the bot.

`test.py` runs the bot on a dataset and returns cash held.

`visualisation.py` contains helpers for visualising the bot and the algorithms. (note might move this alongisde the bot and algos)