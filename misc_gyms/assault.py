import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import ale_py
from enum import Enum
from collections import namedtuple, deque, defaultdict
import random
from gymnasium.wrappers import ResizeObservation, FrameStackObservation


gym.register_envs(ale_py)
torch.set_default_device("mps")
losses = []
q_values = []


class Action(Enum):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    RIGHTFIRE = 5
    LEFTFIRE = 6


N_EPISODES = 6000
MAX_STEPS = 3_000
MEMORY_SIZE = 4096
GAMMA = 0.99
LEARNING_RATE = 3e-3
TAU = 0.005
MIN_EPSILON = 0.001


N_ACTIONS = 7


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    def __init__(self):
        self.memory = deque([], maxlen=MEMORY_SIZE)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, n):
        return Transition(*zip(*random.sample(self.memory, n)))

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Feature Extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        # Fully Connected Layers (Linear)
        # 3136 is derived from the CNN output shape (64 * 7 * 7)
        self.fc = nn.Sequential(
            nn.Linear(2560, 1024), nn.ReLU(), nn.Linear(1024, N_ACTIONS)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_action(self, env, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                # This just gets the indicies of the highest q-value action. `t.max` just makes it a bit weird
                q_values.append(self(state).max(1).values.item())
                return self(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

    def load_weights(self, other: nn.Module):
        target_net_state_dict = other.state_dict()
        q_net_state_dict = self.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        other.load_state_dict(target_net_state_dict)


def decay_epsilon(e, total_t):
    return e * 0.999994


def batch_update(env, mem, qnet: nn.Module, tnet, optimiser):
    batch_sz = 2048
    if len(mem) < batch_sz:
        return
    batch = mem.sample(batch_sz)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
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
        next_state_vals[non_final_mask] = tnet(non_final_next_states).max(1).values

    expected_state_action_vals = (next_state_vals * GAMMA) + reward_batch

    c = nn.SmoothL1Loss()
    loss = c(state_action_vals, expected_state_action_vals.unsqueeze(1))
    if loss is not None:
        losses.append(loss.item())
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(qnet.parameters(), 100)
    optimiser.step()


def main():
    env = gym.make("ALE/Assault-v5", obs_type="grayscale")
    env = ResizeObservation(env, (84, 64))
    env = FrameStackObservation(env, 4)

    mem = ReplayMemory()
    epsilon = 1.0

    net = Network()
    target_net = Network()
    optimiser = AdamW(net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    rewards = []

    total_t = 0
    training = True
    for ep in range(N_EPISODES):
        if ep == int(N_EPISODES * 0.8):
            env = gym.make("ALE/Assault-v5", render_mode="human", obs_type="grayscale")
            env = ResizeObservation(env, (84, 64))
            env = FrameStackObservation(env, 4)
            training = False
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in range(MAX_STEPS):
            # TODO: Implement action masking for only valid actions
            action = net.get_action(env, state, epsilon)

            obs, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward])
            rewards.append(reward.item())

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            mem.add(state, action, reward, next_state, terminated or truncated)
            state = next_state

            total_t += 1
            epsilon = max(
                MIN_EPSILON,
                decay_epsilon(epsilon, total_t),
            )
            if len(losses) > 100 and len(rewards) > 250 and len(q_values) > 250:
                print(
                    f"[{ep}]: e={epsilon:.8f} loss={sum(losses[-100:]) / len(losses[-100:])} reward={sum(rewards[-250:]) / len(rewards[-250:]):.6f} q_val_avg={sum(q_values[-250:]) / len(q_values[-250:]):.6f}",
                    end="\r",
                )
            else:
                print(f"[{ep}]: e={epsilon:.8f}\t\t\t\t\t\t", end="\r")
            if total_t % 256 == 0 and training:
                batch_update(env, mem, net, target_net, optimiser)
            if total_t % 512 == 0:
                net.load_weights(target_net)
            if terminated or truncated:
                break


main()
