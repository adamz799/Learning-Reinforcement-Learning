import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../")
from collections import defaultdict
from env.WindyGridworld import WindyGridWorldEnv

env = WindyGridWorldEnv()
SHAPE = env.shape.append(env.nA)
policy = np.zeros(SHAPE)+1.0/env.nA #random policy
#prob_sum = np.sum(policy,axis=len(SHAPE)-1)[:,:,np.newaxis]


def act(env, policy, state, action):   
    next_state, reward, is_done = env.step(state, action)
    next_action = np.random.choice(env.nA, p = policy[next_state])
    return  reward, next_state, next_action, is_done 
     

def SARSA(env, policy, num_episodes=10000, alpha = 1e-3, gamma = 0.95, epsilon = 0.2):
    """
    Sarsa algorithm. On-policy TD control for esimating Q-value.
    
    Args:
        policy: A numpy array that maps a state-action pair to probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes.
        alpha: Step size.
        gamma: Discount factor.
        epsilon: Factor for epsilon-greedy policy
    
    Returns:
        A numpy array that maps from (state,action) -> value.
        The state-action pair is a tuple and the value is a float.
    """

    Q = np.random.random(SHAPE)
    Q[env.termination]=0.0

    for _ in range(num_episodes):
        state = env.init_state # S
        action = np.random.choice(env.nA, p = policy[state]) # A
        while 1:            
            reward, next_s, next_a, is_done = act(env, policy, state, action) # R S A       
            s_a = state+(action,) #state_action pair 
            next_s_a = next_s+(next_a,)   
            
            Q[s_a] += alpha*(reward+gamma*Q[next_s_a]-Q[s_a])            
            
            #policy_improvement
            greedy_action = np.argmax(Q[state])               
            p = epsilon/env.nA
            for a in range(env.nA):
                policy[state+(a,)] = 1.0-epsilon+p if a is greedy_action else p
            
            if is_done: #next_s is termination
                break
            else:
                state = next_s
                action = next_a
    return Q
