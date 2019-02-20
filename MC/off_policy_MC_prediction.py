import numpy as np
import sys
import math
if "../" not in sys.path:
  sys.path.append("../")
from collections import defaultdict
from env.Blackjack import BlackjackEnv


env = BlackjackEnv()
#shape = [sum of player, dealer's first card, usable_ace, amount of actions]
SHAPE = [sub_space.n for sub_space in env.observation_space.spaces].append(env.nA)
target_policy = np.zeros(SHAPE)+1.0/env.nA#random policy
behavior_policy = np.zeros(SHAPE)+1.0/env.nA

# policy = defaultdict(lambda: 1.0/env.nA)

def act(state, policy):
    """
    state = (player_score, dealer_first_card, usable_ace)
    """
    if policy == None: #for sample policy
        return 0 if state[0] >= 20 else 1 
    else: #for other policy       
        P = policy[state] #A tuple can be used as an index of numpy array
        return np.random.choice(np.arange(env.nA), p=P)

def generate_episodes(policy, env):
    """
    step = (state, action, reward)
    """
    episode = []
    done = False
    state = env.reset() #reset firstly to reset env and get initial state
    while not done:
        action = act(state, policy)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    
    return episode

def off_policy_MC(env, pi = None, b = None, num_episodes=10000, gamma = 0.95):
    """
    Every-visit Off-policy Estimation of a Blackjack State Value with weighted importance sampling

    Args:
        env: OpenAI gym environment.
        pi: Target policy. 
        b: Behavior policy        
        num_episodes: Number of episodes.
        gamma: Discount factor.
    
    Returns:
        A numpy array that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # returns_avg = defaultdict(float)
    # returns_count = defaultdict(float)
    # V = defaultdict(float)
    ratio_sum = np.zeros(SHAPE)
    Q = np.zeros(SHAPE)
    
    for _ in range(num_episodes):
        episode = generate_episodes(b, env)
        G = 0
        ratio = 1.0
        met_states = {}
        for state, action, reward in episode[::-1]:
            if not math.isclose(ratio, 0.0):
                s_a = state+(action,)
                G = gamma*G + reward      
                ratio_sum[s_a]+=ratio    
                Q[state] += (G-Q[s_a])*ratio/ratio_sum[s_a]
                ratio *= pi[s_a]/b[s_a]
                met_states[state] = False
    
    return Q