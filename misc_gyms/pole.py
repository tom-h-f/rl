import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from collections import namedtuple, deque, defaultdict
import numpy as np
import random

N_OF_EPISODES = 10_500
MEMORY_SIZE = 4096
EPSILON_DECAY_RATE = 0.9995
GAMMA = 0.99
LEARNING_RATE = 3e-4
MAX_STEPS = 200
TAU = 0.005
MIN_EPSILON = 0.001

N_OBSERVATIONS = 4
N_ACTIONS = 2


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(N_OBSERVATIONS, 256)
        self.h1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, N_ACTIONS)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.h1(s))
        return self.l2(s)

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                # This just gets the indicies of the highest q-value action. `t.max` just makes it a bit weird
                return self(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

    def load_weights(self, other: nn.Module):
        target_net_state_dict = other.state_dict()
        q_net_state_dict = self.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        other.load_state_dict(target_net_state_dict)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory():
    def __init__(self):
        self.memory = deque([], maxlen=MEMORY_SIZE)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, n):
        return Transition(*zip(*random.sample(self.memory, n)))

    def __len__(self):
        return len(self.memory)


def decay_epsilon(e):
    return e * EPSILON_DECAY_RATE


def batch_update(env, mem, qnet: nn.Module, tnet, optimiser):
    batch_sz = 2048
    if len(mem) < batch_sz:
        return
    batch = mem.sample(batch_sz)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # This computes Q(s_t, a) according to the q_network
    state_action_vals = qnet(state_batch).gather(1, action_batch)

    # then, we compute V(s_(t+1)) for all next states
    # according to the old target_network
    # This lets us have all the values required to calculate the target and error for Q-learning
    # We either have the next state value or 0 if the state was final.
    next_state_vals = torch.zeros(batch_sz)

    with torch.no_grad():
        next_state_vals[non_final_mask] = tnet(
            non_final_next_states).max(1).values

    expected_state_action_vals = (next_state_vals * GAMMA) + reward_batch

    c = nn.SmoothL1Loss()
    loss = c(state_action_vals, expected_state_action_vals.unsqueeze(1))
    print(f"loss={loss:.9f}", end="\r")
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(qnet.parameters(), 100)
    optimiser.step()


def main():
    global env
    epsilon = 1.0
    mem = ReplayMemory()

    q_network = QNetwork()
    target_network = QNetwork()
    optimiser = AdamW(q_network.parameters(), lr=LEARNING_RATE, amsgrad=True)

    print(np.shape(env.action_space))
    for ep in range(N_OF_EPISODES):
        if ep == 8000:
            env = gym.make("CartPole-v1", render_mode="human")
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in range(MAX_STEPS):
            action = q_network.get_action(state, epsilon)
            obs, reward, \
                terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward])

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    obs, dtype=torch.float32).unsqueeze(0)

            mem.add(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if t % 32 == 0:
                print(f"\t\t\t\t[{ep}]: e={epsilon}", end="\r")
                batch_update(env, mem, q_network, target_network, optimiser)
                q_network.load_weights(target_network)
            if terminated or truncated:
                break

        epsilon = max(MIN_EPSILON, decay_epsilon(epsilon))


# torch.set_default_device("mps")
env = gym.make("CartPole-v1")
main()
