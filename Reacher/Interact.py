from collections import deque
import numpy as np
import torch

def Training(agent,env,brain_name,num_agents, num_episodes=200, max_t = 1000, window=100):
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
    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        agent.reset()
        #agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        #agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
        #agent.actor_target.load_state_dict(torch.load('checkpoint_actor_target.pth'))
        #agent.critic_target.load_state_dict(torch.load('checkpoint_critic_target.pth'))                                         
        for t in range(max_t):
            actions = agent.act(states)   
            env_info = env.step(actions)[brain_name]      
            next_states = env_info.vector_observations  
            rewards = env_info.rewards               
            dones = env_info.local_done
            score += rewards
            for i_agent in range(num_agents): 
                agent.step(states[i_agent],actions[i_agent],rewards[i_agent],next_states[i_agent],dones[i_agent])                                       
            states = next_states                            
            if np.any(dones):                                     
                break
        score_mean =  np.mean(score)
        scores_window.append(score_mean)       
        scores.append(score)            
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window),score_mean, end=""))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_target.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_target.pth')
    return scores
    

def Evaluation(agent,env,brain_name):
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    env_info = env.reset(train_mode=False)[brain_name]                # reset the environment
    state = env_info.vector_observations[0]                           # get the current state
    score = 0                                                         # initialize the score
    while True:
        action = agent.act(state, add_noise= False)
        action =  np.clip(action, -1, 1)                             
        env_info = env.step(action)[brain_name]                       # send the action to the environment
        next_state = env_info.vector_observations[0]                  # get the next state
        reward = env_info.rewards[0]                                  # get the reward
        done = env_info.local_done[0]                                 # see if episode has finished
        score += reward                                               # update the score
        state = next_state                                            # roll over the state to next time step
        if done:                                                      # exit loop if episode finished
            break
    return score
    