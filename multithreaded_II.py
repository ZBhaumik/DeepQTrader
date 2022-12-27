"""
Script for training Stock Trading Bot.
Usage:
  multithreaded_II.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--init-episode=<init-episode>] [--pretrained] 

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
  --init-episode=<init-episode>     An initial episode for pausing/resuming training. [default: 1]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
"""
#UTILS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import logging
import random

import pandas as pd
import numpy as np

from tqdm import tqdm
import yfinance as yf
import keras.backend as K
from operations import get_state


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                    .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                    .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


dates = ["2016-08-01","2017-01-01","2017-01-02","2018-01-02","2018-01-03","2019-01-03"]
def yfinance_retrieve(stock_name, type):
    type=type*2
    df = yf.download(stock_name, start=dates[type], end=dates[type+1])
    return list(df['Adj Close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#OPS
import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

#METHODS

import os
import logging
import numpy as np
from collections import deque
import dill as pickle

from global_fns import act
import multiprocess as mp

from parallel_processing import process_chunks

def train_model(agent, episode, data, initial_offset, ep_count=100, batch_size=32, window_size=10, chunk_size=None):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = [0] * data_length

    for t in range(data_length):
        bought_price=0
        state = get_state(data, t, window_size + 1)
        next_state = get_state(data, t + 1, window_size + 1)
        action = agent.act(state)

        # Buy
        if action == 1:  # buy
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
            else:
                agent.inventory.append(data[t])
        # Sell
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            total_profit += data[t] - bought_price
        agent.memory.append((state, action, data[t] - bought_price, next_state, 0))

        if (t == data_length - 1) and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            total_profit += data[t] - bought_price

        # Update model weights every "batch_size" iterations
        if t % batch_size == 0:
            agent.train_experience_replay(batch_size)

    if chunk_size is not None:
        # Split data into chunks
        chunks = [(i * chunk_size, i * chunk_size + chunk_size) for i in range(int(len(data) / chunk_size))]
        chunks[-1] = (chunks[-1][0], len(data))  # handle last chunk which might be smaller than chunk_size

        # Process data chunks in parallel
        avg_loss = process_chunks(data, window_size, batch_size, chunks, agent)

    # Calculate final profit
    final_profit = total_profit + initial_offset
    show_train_result((episode + 1, ep_count, final_profit, avg_loss), 0, initial_offset)

import multiprocess as mp

def evaluate_model_worker(agent, data, window_size, debug, result_queue):
    total_profit = 0
    agent.inventory = []
    state = get_state(data, 0, window_size + 1)
    for t in range(len(data) - 1):
        action = agent.act(state)
    # BUY
    if action == 1:
        agent.inventory.append(data[t])

    # SELL
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price

    if debug:
        print(f"{data[t]}:{state}:{action}:{total_profit}")

    next_state = get_state(data, t + 1, window_size + 1)
    state = next_state
    result_queue.put(total_profit)

def evaluate_model(agent, data, window_size=10, debug=False):
    """Evaluates the agent's performance on the given data
    """
    result_queue = mp.Queue()
    processes = []
    for i in range(4):
        start_index = int(i * len(data) / 4)
        end_index = int((i + 1) * len(data) / 4)
        p = mp.Process(target=evaluate_model_worker, args=(agent, data[start_index:end_index], window_size, debug, result_queue))
        processes.append(p)
        p.start()
    total_profit = 0
    for i in range(4):
        total_profit += result_queue.get()
    for p in processes:
        p.join()

    return total_profit


#AGENT
import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        state = np.reshape(state, (1, self.state_size))
        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        if len(self.memory) < batch_size or batch_size < 0:
            batch_size = len(self.memory)  # set batch_size to the size of the memory attribute
        # Sample a mini-batch from memory
        mini_batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert mini-batch elements to Numpy arrays
        states = np.array(states)
        states = np.reshape(states, (-1, 10))
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        next_states = np.reshape(next_states, (-1, 10))
        dones = np.array(dones, dtype=np.uint8)

        # DQN with fixed targets
        if self.strategy == "t-dqn":
            # Reset target model weights every "reset_every" iterations
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            # Calculate targets
            targets = rewards + np.where(dones, 0, self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1))

            # Predict Q-values for current states
            q_values = self.model.predict(states, verbose=0)

            # Update targets for the actions taken in the mini-batch
            q_values[range(batch_size), actions] = targets

        # Double DQN
        elif self.strategy == "double-dqn":
            # Reset target model weights every "reset_every" iterations
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            # Predict Q-values for current states
            q_values = self.model.predict(states)

            # Calculate targets
            targets = rewards + np.where(dones, 0, self.gamma * self.target_model.predict(next_states, verbose=0)[range(batch_size), np.argmax(self.model.predict(next_states, verbose=0), axis=1)])

            # Update targets for the actions taken in the mini-batch
            q_values[range(batch_size), actions] = targets

        else:
            raise NotImplementedError()

        # Update model weights based on huber loss gradient
        loss = self.model.fit(states, q_values, epochs=1, verbose=0).history["loss"][0]

        # Decrease epsilon to make the agent make more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)

#TRAIN
import logging
import coloredlogs
from docopt import docopt

def main(train_stock, val_stock, window_size=10, batch_size=32, ep_count=10,
        strategy="t-dqn", model_name="model_debug", pretrained=False,
        debug=False, init_episode=1):
        agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
        train_data = yfinance_retrieve(train_stock, 0)
        val_data = yfinance_retrieve(val_stock, 1)

        initial_offset = val_data[1] - val_data[0]
        for episode in range(init_episode, ep_count + 1):
            print(episode)
            train_result = train_model(agent, episode, train_data, initial_offset=initial_offset, ep_count=ep_count,
                                    batch_size=batch_size, window_size=window_size, chunk_size=1)
            val_result = evaluate_model(agent, val_data, window_size, debug)
            show_train_result(train_result, val_result, initial_offset)

if __name__ == "__main__":
    print("HELLO")
    args = docopt(__doc__)
    print("HI")
    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    init_episode = int(args["--init-episode"])
    pretrained = args["--pretrained"]
    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size=window_size, batch_size=batch_size,
             ep_count=ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, init_episode=init_episode)
    except KeyboardInterrupt:
        print("Aborted!")