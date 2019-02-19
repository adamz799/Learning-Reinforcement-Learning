import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../")
from collections import defaultdict
from env.Blackjack import BlackjackEnv
from MC_prediction import act, generate_episodes


env = BlackjackEnv()
random_policy = defaultdict(lambda: 1.0/env.nA)

def MC_estimating(env, policy = None, num_episodes=10000, gamma = 0.95, epsilon = 0.2):
    """
    Monte Carlo estimating algorithm. Calculates the Q function for a given 
    policy using sampling.
    
    Args:
        policy: A dictionary that maps a (state,action) to probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes.
        gamma: discount factor.
        epsilon: factor for ε-greedy policy
    
    Returns:
        A dictionary that maps from (state,action) -> value.
        The state-action pair is a tuple and the value is a float.
    """

    #returns_avg = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(float) 
    
    for _ in range(num_episodes):
        episode = generate_episodes(policy, env)
        G = 0
        met_sa = []
        for state, action, reward in episode[::-1]:
            G = gamma*G + reward
            s_a = (state, action)#state_action pair
            if state not in met_sa:#first-visit MC estimating
                Q[s_a] += (G-Q[s_a])/(returns_count[state]+1)
                returns_count[s_a]+=1
                met_sa.append(s_a)

                #policy_improvement
                actions_prob = np.ndarray(env.nA)
                for a in range(env.nA):
                    actions_prob[a] = policy[(state,a)]
                greedy_action = np.argmax(actions_prob)
                
                p = epsilon/env.nA
                for a in range(env.nA):
                    policy[(state,a)] = 1.0-epsilon+p if a is greedy_action else p

    return Q
