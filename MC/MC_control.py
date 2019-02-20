import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../")
from collections import defaultdict
from env.Blackjack import BlackjackEnv
from MC_prediction import act, generate_episodes


env = BlackjackEnv()
SHAPE = [sub_space.n for sub_space in env.observation_space.spaces].append(env.nA)
policy = np.zeros(SHAPE)+1.0/env.nA #random policy

def MC_estimating(env, policy = None, num_episodes=10000, gamma = 0.95, epsilon = 0.2):
    """
    Monte Carlo estimating algorithm. Calculates the Q function for a given 
    policy using sampling.
    
    Args:
        policy: A numpy array that maps a state-action pair to probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes.
        gamma: Discount factor.
        epsilon: Factor for ε-greedy policy
    
    Returns:
        A numpy array that maps from (state,action) -> value.
        The state-action pair is a tuple and the value is a float.
    """

    #returns_avg = defaultdict(float)
    returns_count = np.zeros(SHAPE)
    Q = np.zeros(SHAPE)
    
    for _ in range(num_episodes):
        episode = generate_episodes(policy, env)
        G = 0
        met_sa = {}
        for state, action, reward in episode[::-1]:
            G = gamma*G + reward
            s_a = state+(action,) #state_action pair    
            #if met_sa.get(s_a,Ture):       
            if s_a not in met_sa: #first-visit MC estimating
                Q[s_a] += (G-Q[s_a])/(returns_count[s_a]+1)
                returns_count[s_a]+=1
                met_sa[s_a] = False

                #policy_improvement
                greedy_action = np.argmax(Q[state])               
                p = epsilon/env.nA
                for a in range(env.nA):
                    policy[s_a] = 1.0-epsilon+p if a is greedy_action else p

    return Q
