import gymnasium as gym
import torch
import numpy as np
import random

learning_rate = 0.1
gamma = 0.95
epsilon_decay = 0.975
BASE_RANDOM_SEED = 18974981

N_EPISODES = 1000
MAX_STEPS = 1000
MAX_EPSILON = 1.0

N_BINS = [1, 1, 6, 3]


def main():
    env = gym.make("CartPole-v1")
    print(env.observation_space.shape)
    q_train(env)


def create_bins(env):
    obs_low = env.observation_space.low.astype(np.float64)
    obs_high = env.observation_space.high.astype(np.float64)

    # Clip infinite / huge ranges (CartPole-specific)
    obs_low[1] = -3.0
    obs_high[1] = 3.0
    obs_low[3] = -3.0
    obs_high[3] = 3.0

    bins = []
    for i, n in enumerate(N_BINS):
        if n == 1:
            bins.append(None)  # no binning, single bucket
        else:
            # N_BINS[i] bins => N_BINS[i]-1 boundaries
            bins.append(np.linspace(obs_low[i], obs_high[i], n - 1))
    return bins, obs_low, obs_high


def discretize_state(state: np.ndarray,
                     bins,
                     low: np.ndarray,
                     high: np.ndarray) -> tuple[int, ...]:
    idxs = []
    for i, n in enumerate(N_BINS):
        if n == 1:
            idxs.append(0)
            continue

        # Clip state within [low, high] for this dim
        v = np.clip(state[i], low[i], high[i])

        # digitize: which bin does v fall into?
        # bins[i] is a 1D array of thresholds, sorted increasing
        idx = np.digitize(v, bins[i])

        # ensure idx in [0, n-1]
        idx = min(max(idx, 0), n - 1)

        idxs.append(idx)

    return tuple(idxs)  # e.g. (0, 0, 3, 1)


def q_train(env: gym.Env):
    np.random.seed(BASE_RANDOM_SEED)
    random.seed(BASE_RANDOM_SEED)

    bins, low, high = create_bins(env)

    # init Q-table
    q_table = np.zeros((*N_BINS, 2))

    t = 0
    print(q_table)
    epsilon = 0.99
    for episode in range(N_EPISODES):

        state, info = env.reset(seed=BASE_RANDOM_SEED)
        for step in range(MAX_STEPS):
            action = get_action(q_table, state, bins, low, high, t, epsilon)
            # Take action, observe reward and state
            next_state, reward, terminated, truncated, info = env.step(action)

            disc_state = discretize_state(state, bins, low, high)
            # new_qsa = qsa + (learning_rate*(reward + (gamma * argmax(Q(s+1, a)))))
            former_estimation = q_table[disc_state][action]

            next_state_max_q = get_action(q_table, next_state, bins,
                                          low, high, t, epsilon)
            q_table[disc_state][action] = former_estimation + \
                (learning_rate*(reward + (gamma * next_state_max_q) - former_estimation))
            print(f"action={action} reward={reward} q_value={
                  former_estimation} new_q_value={q_table[disc_state][action]} state={state}", end="\r")

            if terminated or truncated:
                break

            t += 1
        print(f"EPISODE {episode} DONE")
    env = gym.make("CartPole-v1", render_mode="human")
    while True:
        state, info = env.reset(seed=BASE_RANDOM_SEED)
        for step in range(MAX_STEPS):
            action = get_action(q_table, state, bins, low, high, t, epsilon)
            # Take action, observe reward and state
            next_state, reward, terminated, truncated, info = env.step(action)

            disc_state = discretize_state(state, bins, low, high)
            # new_qsa = qsa + (learning_rate*(reward + (gamma * argmax(Q(s+1, a)))))
            former_estimation = q_table[disc_state][action]

            next_state_max_q = get_action(q_table, next_state, bins,
                                          low, high, t, epsilon)
            q_table[disc_state][action] = former_estimation + \
                (learning_rate*(reward + (gamma * next_state_max_q) - former_estimation))
            print(f"action={action} reward={reward} q_value={
                  former_estimation} new_q_value={q_table[disc_state][action]} state={state}", end="\r")

            if terminated or truncated:
                break


def get_action(q_table: np.ndarray, state, bins, low, high, t, e) -> int:
    r = random.uniform(0.0, 1.0)
    disc_state = discretize_state(state, bins, low, high)
    q_actions = q_table[disc_state]
    e *= epsilon_decay
    if r < e:
        return int(np.random.randint(0, len(q_actions)))
    else:
        return int(np.argmax(q_actions))


if __name__ == "__main__":
    main()
