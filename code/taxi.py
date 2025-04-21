import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from pathlib import Path
from tqdm import tqdm

# Q‑learning hyperparameters
alpha, gamma, epsilon = 0.1, 0.99, 0.1
episodes = 10_000

# where to save videos
VIDEO_DIR = Path(__file__).resolve().parent.parent / "notes" / "_media"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# create and wrap the environment, recording only on episode 1 and the final episode
env = gym.make("Taxi-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=str(VIDEO_DIR),
    name_prefix="taxi_q_learning",
    episode_trigger=lambda ep: ep == 0 or ep == episodes - 1
)

n_states, n_actions = env.observation_space.n, env.action_space.n
Q = np.zeros((n_states, n_actions))

for ep in tqdm(range(episodes), desc="Training episodes"):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q‑learning update
        td = reward + gamma * np.max(Q[next_state]) - Q[state, action]
        Q[state, action] += alpha * td
        state = next_state

env.close()
