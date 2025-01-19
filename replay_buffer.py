from collections import deque
import random
from torch import cat

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def __len__(self):
        return len(self.memory)
    
    def append(self, transition):
        self.memory.append(transition)

    def can_sample(self, batch_size):
        return len(self.memory) > batch_size
    
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [cat(item) for item in batch]
