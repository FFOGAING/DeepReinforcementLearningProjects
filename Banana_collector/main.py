from unityagents import UnityEnvironment
from monitor import interact
from agent import Agent
import numpy as np
#Start  a Unity environment
#Give as file_name the  path of your Unity environment like "Banana_Windows_x86_64/Banana.exe"
env = UnityEnvironment(file_name="...")

# get the default brain that will be control to the python code
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# Create agent instance
agent= Agent()
# interact with the environment
avg_rewards, best_avg_reward = interact(env, agent)

# close the environment when finished
env.close()
