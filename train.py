import torch
import gymnasium as gym
from noise import OrnsteinUhlenbeckNoise
from ddpg import DDPG
import numpy as np
from replay_buffer import ReplayBuffer

env = gym.make("Ant-v5")
state_dims, action_dims = env.observation_space.shape[0], env.action_space.shape[0]
h1 = 400
h2 = 300
tau = 0.05
gamma = 0.99
epoch=0
episodes=400
batch_size=128
ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_dims))
memory = ReplayBuffer(1_000_000)
agent = DDPG(state_dims=state_dims, action_dims=action_dims, h1=h1, h2=h2, tau=tau, gamma=gamma)

for episode in range(1, episodes+1):
    state, _ = env.reset()
    state = torch.tensor(state).unsqueeze(0).float()
    done=False
    timesteps=0
    episode_return = 0
    while not done:
        timesteps+=1
        action = agent.chooose_action(state, ou_noise).numpy()[0]
        next_state, reward, done, trunc, info = env.step(action)
        episode_return+=reward
        next_state = torch.tensor(next_state).unsqueeze(0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        action = torch.tensor(action).view(1, -1).float()
        memory.append([state, action, reward, done, next_state])
        state = next_state

        epoch_critic_loss = 0
        epoch_actor_loss = 0

        if memory.can_sample(batch_size):
            batch = memory.sample(batch_size)
            critic_loss, actor_loss = agent.update_params(batch)
            epoch_critic_loss+=critic_loss
            epoch_actor_loss+=actor_loss
        
        if trunc:
            print(f"Episode Truncated at {timesteps}")
            break
    
    if episode % 50 == 0:
        torch.save(agent.actor.state_dict(), f"models\ddpg_ant_{episode}.pt")
        print(f"...... Model for episode {episode} is saved ......")

    
    print(f"for episode {episode} return {episode_return:.2f} critic loss {epoch_critic_loss:.2f} actor loss {epoch_actor_loss:.2f}")







