import os
import logging
import numpy as np
from tqdm import tqdm
from utilities import format_currency, format_position
from operations import get_state

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1
    agent.inventory = []
    avg_loss = []
    state = get_state(data, 0, window_size + 1)
    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        action = agent.act(state)
        if action == 1:
            agent.inventory.append(data[t])
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta
        else:
            pass
        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)
        state = next_state
    if episode % 10 == 0:
        agent.save(episode)
    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1
    history = []
    agent.inventory = []
    state = get_state(data, 0, window_size + 1)
    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        action = agent.act(state, is_eval=True)
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        else:
            history.append((data[t], "HOLD"))
        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return total_profit, history