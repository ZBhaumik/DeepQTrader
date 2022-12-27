import multiprocess as mp
import numpy as np
from collections import deque
from operations import get_state
from global_fns import exp_replay, act

def process_chunks(data, window_size, batch_size, chunks, agent):
    # Process data chunks in parallel
    p = mp.Pool()
    results = p.map(process_chunk, [(data, window_size, batch_size, chunk, act, exp_replay, agent) for chunk in chunks])

    # Calculate average loss for the episode
    avg_loss = np.mean(results)

    # Close the pool
    p.close()
    p.join()
    
    return avg_loss

def process_chunk(args):
    data, window_size, batch_size, chunk, act, exp_replay, agent = args

    # Initialize variables
    n_iter = 0
    total_loss = 0
    inventory = []
    memory = deque(maxlen=2000)
    initial_offset = 0.0
    for t in range(chunk[0], chunk[1]):
        state = get_state(data, t, window_size + 1)
        action = act(agent, state)

        # Buy
        if action == 1:
            agent.inventory.append(data[t])
            inventory.append(data[t])
        # Sell
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            initial_offset += data[t] - bought_price
            total_profit += data[t] - bought_price
            inventory.append(data[t])
        # Hold
        elif action == 0:
            inventory.append(0)

        if t == chunk[1] - 1:
            bought_price = agent.inventory.pop(0)
            initial_offset += data[t] - bought_price
            total_profit += data[t] - bought_price
            inventory.append(data[t])

        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0
        if t == chunk[1] - 1:
            reward = initial_offset
        memory.append((state, action, data[t] - bought_price, next_state, reward))

        # Update model weights every "batch_size" iterations
        if t % batch_size == 0:
            loss = exp_replay(agent, batch_size, memory)
            total_loss += loss
            n_iter += 1

    return total_loss / n_iter