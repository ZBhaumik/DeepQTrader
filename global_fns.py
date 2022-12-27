import os
import logging
import numpy as np
from collections import deque
from multiprocessing import Pool
import random

def act(agent, state, is_eval=False):
    """Take action from given possible set of actions
    """
    # take random action in order to diversify experience at the beginning
    if not is_eval and random.random() <= agent.epsilon:
        return random.randrange(agent.action_size)

    if agent.first_iter:
        agent.first_iter = False
        return 1  # make a definite buy on the first iter

    state = np.reshape(state, (1, agent.state_size))
    action_probs = agent.model.predict(state, verbose=0)
    return np.argmax(action_probs[0])

def exp_replay(agent, batch_size):
    """Train on previous experiences in memory
    """
    # Sample a mini-batch from memory
    mini_batch = random.sample(agent.memory, batch_size)
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
    if agent.strategy == "t-dqn":
        # Reset target model weights every "reset_every" iterations
        if agent.n_iter % agent.reset_every == 0:
            agent.target_model.set_weights(agent.model.get_weights())

        # Calculate targets
        targets = rewards + np.where(dones, 0, agent.gamma * np.amax(agent.target_model.predict(next_states, verbose=0), axis=1))

        # Predict Q-values for current states
        q_values = agent.model.predict(states, verbose=0)

        # Update targets for the actions taken in the mini-batch
        q_values[range(batch_size), actions] = targets

    # Double DQN
    elif agent.strategy == "double-dqn":
        # Reset target model weights every "reset_every" iterations
        if agent.n_iter % agent.reset_every == 0:
            agent.target_model.set_weights(agent.model.get_weights())

        # Predict Q-values for current states
        q_values = agent.model.predict(states)

        # Calculate targets
        targets = rewards + np.where(dones, 0, agent.gamma * agent.target_model.predict(next_states, verbose=0)[range(batch_size), np.argmax(agent.model.predict(next_states, verbose=0), axis=1)])

        # Update targets for the actions taken in the mini-batch
        q_values[range(batch_size), actions] = targets

    else:
        raise NotImplementedError()

    # Update model weights based on huber loss gradient
    loss = agent.model.fit(states, q_values, epochs=1, verbose=0).history["loss"][0]
    # Decrease epsilon to make the agent make more optimal decisions
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    return loss