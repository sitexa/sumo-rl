import sys
import traci
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
# import gym_sumo
from gym import wrappers

# Define the environment
env = gym.make("sumo-simple-grid-v0")

# Define the Q-Learning algorithm
q_table = {}

# Define the training parameters
num_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 0.1
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Train the Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Choose an action
        if random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        # Take the action and observe the new state and reward
        next_state, reward, done, info = env.step(action)
        rewards_current_episode += reward

        # Update the Q-value for the new state-action pair
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (reward + discount_rate * np.max(q_table[next_state]))

        # Update the state
        state = next_state

        # End the episode if it is done
        if done:
            break

    # Decrease the exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# Test the trained Q-Learning algorithm
total_rewards = []
for episode in range(100):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state])
        next_state, reward, done, info = env.step(action)
        rewards_current_episode += reward
        state = next_state
        if done:
            break
    total_rewards.append(rewards_current_episode)

print("Average reward: ", sum(total_rewards) / len(total_rewards))

# Close the environment
env.close()
