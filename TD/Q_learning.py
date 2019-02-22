import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../")
from collections import defaultdict
from env.CliffWalkingEnv import CliffWalkingEnv

env = CliffWalkingEnv()
SHAPE = env.shape.append(env.nA)
policy = np.zeros(SHAPE)+1.0/env.nA #random policy
#prob_sum = np.sum(policy,axis=len(SHAPE)-1)[:,:,np.newaxis]


def act_Q(env, state, action):   
    next_state, reward, is_done = env.step(state, action)   
    return  reward, next_state, is_done 
     

def Q_learning(env, policy, num_episodes=10000, alpha = 1e-3, gamma = 0.95, epsilon = 0.2):
    """
    Q_learning algorithm. Off-policy TD Control.
    
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
        state = env.init_state 
        
        while 1:            
            action = np.random.choice(env.nA, p = policy[state]) # A
            reward, next_s, is_done = act_Q(env, state, action) # R S A       
            s_a = state+(action,) #state_action pair 

            Q[s_a] += alpha*(reward+gamma*np.max(Q[next_s])-Q[s_a])            
            
            #policy_improvement
            greedy_action = np.argmax(Q[state])               
            p = epsilon/env.nA
            for a in range(env.nA):
                policy[state+(a,)] = 1.0-epsilon+p if a is greedy_action else p
            
            if is_done: #next_s is the termination
                break
            else:
                state = next_s
    return Q
