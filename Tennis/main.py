from unityagents import UnityEnvironment
from Interact import Training, Evaluation
from multiagent import MultiAgent
import numpy as np
import matplotlib.pyplot as plt

#Start  a Unity environment
#Give as file_name the  path of your Unity environment like "Banana_Windows_x86_64/Banana.exe"
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# get the default brain that will be control to the python code
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Create agent instance
agents= MultiAgent(num_agents,state_size,action_size,num_agents)

# With Train = true , the agent will be train
Train = True

#interaction
if(Train):
    scores = Training(agents,env,brain_name,num_agents)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
else:
    score = Evaluation(agents,env,)
    print("Score: {}".format(score))
    
# close the environment when finished
env.close()
