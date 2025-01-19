from network import ActorNetwork, CriticNetwork
# from replay_buffer import ReplayBuffer
# from noise import OrnsteinUhlenbeckNoise as OUNoise
import torch
from update_params_by import soft_update, hard_update
import torch.nn as nn

class DDPG:
    def __init__(self, state_dims, action_dims, gamma, tau, h1, h2, alpha=1e-4, beta=1e-3):
        self.state_dims = state_dims
        # self.action_dims = action_dims
        self.gamma = gamma
        self.tau = tau

        self.actor = ActorNetwork(state_dims=state_dims, action_dims=action_dims, h1=h1, h2=h2)
        self.actor_target = ActorNetwork(state_dims=state_dims, action_dims=action_dims, h1=h1, h2=h2)

        self.critic = CriticNetwork(state_dims=state_dims, action_dims=action_dims, h1=h1, h2=h2)
        self.critic_target = CriticNetwork(state_dims=state_dims, action_dims=action_dims, h1=h1, h2=h2)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=alpha)
        self.critic_optimzer = torch.optim.AdamW(self.critic.parameters(), lr=beta, weight_decay=1e-2)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def chooose_action(self, state, action_noise):
        self.actor.eval()
        mu = self.actor(state)
        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            noise = torch.tensor(action_noise.noise())
            mu+=noise
    
        return mu
    
    def update_params(self, batch):
        state, action, reward, done, next_state = batch

        next_action_batch = self.actor_target(next_state)
        next_state_action_values = self.critic_target(next_state, next_action_batch.detach())
        expected_values = reward + ~done * self.gamma * next_state_action_values

        self.critic_optimzer.zero_grad()
        state_action = self.critic(state, action)
        value_loss = nn.functional.mse_loss(state_action, expected_values.detach())
        value_loss.backward()
        self.critic_optimzer.step()

        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state, self.actor(state))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()










