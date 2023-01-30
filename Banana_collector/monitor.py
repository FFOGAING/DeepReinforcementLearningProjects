from collections import deque
import sys
import math
import numpy as np

def interact(env,brain_name,agent, Train_mode=False, num_episodes=1800, window=100):
    """ Monitor agent's performance
    Params
    ======
    - env: instance of the Unity environment
    - brain_name: name of the agent which schould be control trough 
    - agent: instance of class Agent (see Agent.py for details)
    - train mode: True for Agent's Training and False for Evaluation
    
    Returns
    ======
    - avg_rewards: deque containing average rewards
    - best_arg_reward: largest value in the avvg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    scores = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # reset the environment
        env_info = env.reset(train_mode=Train_mode)[brain_name]
        # get the current state
        state = env_info.vector_observations[0]
        # initialize the score
        score = 0                                         
        while True:
            # agent selects an action
            action = agent.act(state)
            # send the action to the environment
            env_info = env.step(action)[brain_name] 
            # get the next state       
            next_state = env_info.vector_observations[0]  
            # get the reward 
            reward = env_info.rewards[0] 
            # see if episode has finished                 
            done = env_info.local_done[0] 
            # agent performs internal updates based on sampled experience
            agent.step(state,action,reward,next_state,done)   
            # update the score              
            score += reward    
            # roll over the state to next time step                            
            state = next_state 
            # exit loop if episode finished                           
            if done:
                #save final score
                scores.append(score)                                       
                break 
        if (i_episode % 100 == 0):
            # get average reward over every 100 consecutives episodes 
            # ToDO: Schould I reset after every 100 consecutives rewards?
            avg_reward = np.mean(scores)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to Udacity's prescription)
        if best_avg_reward >= 13:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward