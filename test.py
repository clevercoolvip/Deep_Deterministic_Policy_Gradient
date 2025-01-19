import torch
import gymnasium as gym
from network import ActorNetwork
from gymnasium.wrappers import RecordVideo

env = gym.make("Ant-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder='videos/0_episodes', episode_trigger=lambda x: x%1==0)
state_dims, action_dims = env.observation_space.shape[0], env.action_space.shape[0]

# actor = ActorNetwork(state_dims, action_dims, 400, 300)
# actor.load_state_dict(torch.load("models\ddpg_ant_400.pt"))

for episode in range(10):
    state, _ = env.reset()
    state = torch.tensor(state).unsqueeze(0).float()
    done=False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, trunc, info = env.step(action)
        next_state = torch.tensor(next_state).unsqueeze(0).float()
        state=next_state

        if trunc:
            break
