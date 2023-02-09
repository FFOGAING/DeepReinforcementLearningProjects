from collections import deque
import numpy as np
import torch

def Training(agents,env,brain_name, num_agents, num_episodes=5000, max_t = 1000, window=100):
    """ Execute the training of the agent
    Params
    ======
    - agent: instance of class Agent (see Agent.py for details)
    - env: instance of the Unity environment
    - brain_name: name of the agent which schould be control trough 
    - num_episodes: maximum number of training episodes
    - max_t: maximum number of time steps per episodes
    - window: 
    - eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    - eps_end (float): minimum value of epsilon
    - eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    
    Returns
    ======
    - Scores: list containing scores from each episode
    """
    Scores = []                        
    scores_window = deque(maxlen=window)                     
    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agents.reset()                                         
        for t in range(max_t):
            actions = agents.act(states)  
            env_info = env.step(actions)[brain_name]      
            next_states = env_info.vector_observations  
            rewards = env_info.rewards             
            dones = env_info.local_done 
            agents.step(states,actions,rewards,next_states,dones)                
            scores += rewards                                
            states = next_states                            
            if np.any(dones):                                     
                break 
        score = np.max(scores)
        scores_window.append(score)      
        Scores.append(score)              
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window),score, end=""))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agents.save_weights()
    return Scores

def Evaluation(agents,env,brain_name,num_agents):
    agents.load_weights()
    env_info = env.reset(train_mode=False)[brain_name]                # reset the environment
    state = env_info.vector_observations                              # get the current state
    score = np.zeros(num_agents)                                                        # initialize the score
    while True:
        action = agents.act(state, add_noise= False)
        action =  np.clip(action, -1, 1)                             
        env_info = env.step(action)[brain_name]                       # send the action to the environment
        next_state = env_info.vector_observations[0]                  # get the next state
        reward = env_info.rewards[0]                                  # get the reward
        done = env_info.local_done[0]                                 # see if episode has finished
        score += reward                                               # update the score
        state = next_state                                            # roll over the state to next time step
        if done:                                                      # exit loop if episode finished
            break
    return np.max(score)
    