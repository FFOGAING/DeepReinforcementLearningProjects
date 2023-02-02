from collections import deque
import numpy as np
import torch

def Training(agent,env,brain_name, num_episodes=1800, max_t = 1000, window=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
    - scores: list containing scores from each episode
    """
    scores = []                        
    scores_window = deque(maxlen=window)  
    eps = eps_start                    
    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0                                         
        for t in range(max_t):
            action = agent.act(state,eps)
            env_info = env.step(action)[brain_name]      
            next_state = env_info.vector_observations[0]  
            reward = env_info.rewards[0]               
            done = env_info.local_done[0] 
            agent.step(state,action,reward,next_state,done)                
            score += reward                                
            state = next_state                            
            if done:                                     
                break 
        scores_window.append(score)       
        scores.append(score)              
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pt')
            break
    return scores

def Evaluation(agent,env,brain_name):
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pt')) # load the weights from file
    env_info = env.reset(train_mode=False)[brain_name]                # reset the environment
    state = env_info.vector_observations[0]                           # get the current state
    score = 0                                                         # initialize the score
    while True:
        action = agent.act(state)                                     # select an action
        env_info = env.step(action)[brain_name]                       # send the action to the environment
        next_state = env_info.vector_observations[0]                  # get the next state
        reward = env_info.rewards[0]                                  # get the reward
        done = env_info.local_done[0]                                 # see if episode has finished
        score += reward                                               # update the score
        state = next_state                                            # roll over the state to next time step
        if done:                                                      # exit loop if episode finished
            break
    return score
    