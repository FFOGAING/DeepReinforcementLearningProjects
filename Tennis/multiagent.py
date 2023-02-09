import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE =  200      # minibatch size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent():
    """Wrapper of the Agent() class for training of multiples agents."""
    
    def __init__(self,num_agents, state_size, action_size, random_seed=15):
        self.agents = [Agent(state_size,action_size,random_seed) for x in range(num_agents)]
        self.sharememory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    def step(self, states, actions, rewards, next_states, dones):
        
        
        for index, agent in enumerate(self.agents):
            self.sharememory.add(states[index], actions[index], rewards[index], next_states[index], dones[index])
            agent.step(states[index],actions[index],rewards[index],next_states[index],dones[index],self.sharememory)

    def act(self, states, add_noise=True):
        actions = []
        for index, agent in enumerate(self.agents):
            actions.append(agent.act(states[index],add_noise))
        return np.asarray(actions)

    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index+1))
    
    def load_weights(self):
        for index, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('agent{}_checkpoint_actor.pth'.format(index+1)))
            agent.critic_local.load_state_dict(torch.load('agent{}_checkpoint_critic.pth'.format(index+1)))
    def reset(self):        
        for agent in self.agents:
            agent.reset()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)