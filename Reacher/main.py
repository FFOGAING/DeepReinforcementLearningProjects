from unityagents import UnityEnvironment
from Interact import Training, Evaluation
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt

#Start  a Unity environment
#Give as file_name the  path of your Unity environment like "Banana_Windows_x86_64/Banana.exe"
env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

# get the default brain that will be control to the python code
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
num_agents = len(env_info.agents)
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Create agent instance
agent= Agent(state_size,action_size,0)

# With Train = true , the agent will be train
Train = True

#interaction
if(Train):
    scores = Training(agent,env,brain_name,num_agents)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
else:
    score = Evaluation(agent,env)
    print("Score: {}".format(score))
# close the environment when finished
env.close()
