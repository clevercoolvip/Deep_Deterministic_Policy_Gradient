import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

class ActorNetwork(nn.Module):
    def __init__(self, state_dims, action_dims, h1, h2):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dims, h1)
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)
        nn.init.uniform_(self.fc1.bias, -f1, f1)
        self.bn1 = nn.LayerNorm(h1)

        self.fc2 = nn.Linear(h1, h2)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)
        nn.init.uniform_(self.fc2.bias, -f2, f2)
        self.bn2 = nn.LayerNorm(h2)

        f3=0.0003
        self.mu = nn.Linear(h2, action_dims)
        nn.init.uniform_(self.mu.weight, -f3, f3)
        nn.init.uniform_(self.mu.bias, -f3*0.1, f3*0.1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        mu = torch.tanh(self.mu(x))
        return mu
    

class CriticNetwork(nn.Module):
    def __init__(self, state_dims, action_dims, h1, h2):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dims, h1)
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)
        nn.init.uniform_(self.fc1.bias, -f1, f1)
        self.bn1 = nn.LayerNorm(h1)

        self.fc2 = nn.Linear(h1 + action_dims, h2)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)
        nn.init.uniform_(self.fc2.bias, -f2, f2)
        self.bn2 = nn.LayerNorm(h2)

        self.q = nn.Linear(h2, 1)
        f3=0.0003
        nn.init.uniform_(self.q.weight, -f3, f3)
        nn.init.uniform_(self.q.bias, -f3*0.1, f3*0.1)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = torch.cat([x, action], 1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        V = self.q(x)
        return V



if __name__=="__main__":
    env = gym.make("Ant-v5")
    state, _ = env.reset()
    state = torch.tensor(state).unsqueeze(0).float()
    actor = ActorNetwork(105, 8, 400, 300)
    print("Actor", actor(state))
    print("Data", actor(state).data)

    action=torch.tensor(env.action_space.sample()).unsqueeze(0).float()
    critic = CriticNetwork(105, 8, 400, 300)
    print("Critic", critic(state, action))



