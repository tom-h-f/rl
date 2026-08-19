import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import random

torch.set_default_device("mps")

learning_rate = 0.1
n_episodes = 250_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1
gamma = 0.99


class BJ():
    def __init__(self, n_actions: int):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.epsilon = start_epsilon

    def get_action(self, state, env):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, reward, state, action_idx, next_state):
        target = reward + gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] += learning_rate * \
            (target - self.q_table[state][action_idx])


if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=False, render_mode="none")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    bj = BJ(env.action_space.n)
    ep_results = []
    for episode in range(n_episodes):
        state, info = env.reset()
        terminated, truncated = (False, False)
        while not (terminated or truncated):
            # Training Loop
            action = bj.get_action(state, env)
            next_state, reward, terminated, truncated, info = env.step(action)
            bj.update(reward, state, action, next_state)
            state = next_state
        bj.epsilon = max(bj.epsilon - epsilon_decay, final_epsilon)
        ep_results.append(reward)
        print(f"[{episode}] w/d/l: {(ep_results.count(1.0) / len(ep_results))
              * 100:.4f} {(ep_results.count(0.0) / len(ep_results))
              * 100:.4f} {(ep_results.count(-1.0) / len(ep_results))
              * 100:.4f} epsilon={bj.epsilon:.4f}", end="\r")
