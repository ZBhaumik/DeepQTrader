"""
Script for training Stock Trading Bot.
Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]
Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging
import coloredlogs
from docopt import docopt
import sys

from agent import Agent
from methods import train_model, evaluate_model
from utilities import yfinance_retrieve, format_currency, format_position, show_train_result, switch_k_backend_device

def main(train_stock, val_stock, window_size=10, batch_size=32, ep_count=50,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    train_data = yfinance_retrieve(train_stock, 0)
    val_data = yfinance_retrieve(val_stock, 1)
    initial_offset = val_data[1] - val_data[0]
    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count, batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":
    #args = docopt(__doc__)
    train_stock = "GOOGL"#args["<train-stock>"]
    val_stock = "GOOGL"#args["<val-stock>"]
    #strategy = args["--strategy"]
    #window_size = int(args["--window-size"])
    #batch_size = int(args["--batch-size"])
    #ep_count = int(args["--episode-count"])
    #model_name = args["--model-name"]
    #pretrained = args["--pretrained"]
    #debug = args["--debug"]
    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()
    try:
        main(train_stock, val_stock)
    except KeyboardInterrupt:
        print("Aborted!")