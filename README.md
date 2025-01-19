# DDPG for Solving MuJoCo Ant-v5

## Overview
This repository contains the implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the MuJoCo Ant-v5 environment. The DDPG algorithm is an off-policy, model-free reinforcement learning method that operates in continuous action spaces, making it suitable for complex control tasks like the Ant-v5 environment.

## Environment
[MuJoCo](https://mujoco.org/) (Multi-Joint dynamics with Contact) is a physics engine that enables simulation of various robotic and biomechanical systems. The Ant-v5 environment involves controlling a four-legged creature to walk efficiently.

## Implementation
This implementation of DDPG leverages the following key components:
- **Actor-Critic Architecture**: The actor network determines the action to take, and the critic network evaluates the action by estimating the Q-value.
- **Replay Buffer**: Stores experience tuples (state, action, reward, next state) to enable stable training by sampling mini-batches.
- **Target Networks**: Soft updates to target networks help to stabilize training by slowly updating target network parameters towards the learned networks.
- **Ornstein-Uhlenbeck Noise**: Adds exploration noise to the action to ensure sufficient exploration of the action space.

## Dependencies
To run this project, ensure you have the following installed:
- Python 3.9.10
- MuJoCo
- `gymnasium[mujoco]`
- `numpy`
- `pytorch`
- `matplotlib`

Install the dependencies using the following command:
```bash
pip install gymnasium[mujoco] numpy pytorch matplotlib
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ddpg-mujoco-antv5.git
   cd ddpg-mujoco-antv5
   ```
2. Train the agent:
   ```bash
   python train.py
   ```
3. Evaluate the trained agent:
   ```bash
   python test.py
   ```

## File Structure
- `train.py`: Script to train the DDPG agent.
- `test.py`: Script to evaluate the trained DDPG agent.
- `ddpg.py`: Contains the DDPG agent class with the actor-critic networks.
- `replay_buffer.py`: Implementation of the replay buffer.
- `models/`: Directory to save trained models.
- `videos/`: Stored videos of the environment after training

## Results
The trained DDPG agent achieves significant improvement in the performance of the Ant-v5 environment. The learning curve shows the agent's reward increasing over time as it learns to navigate the environment efficiently.

## Contributions
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Contact
For any questions or suggestions, feel free to open an issue or contact me at panda18vishu@gmail.com
