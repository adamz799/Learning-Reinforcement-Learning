import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../")
from collections import defaultdict
from env.Blackjack import BlackjackEnv


env = BlackjackEnv()
#shape = [sum of player, dealer's first card, usable_ace, amount of actions]
SHAPE = [sub_space.n for sub_space in env.observation_space.spaces].append(env.nA)
policy = np.zeros(SHAPE)+1.0/env.nA#random policy

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

def MC_prediction(env, policy = None, num_episodes=10000, gamma = 0.95):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A numpy array that maps a state-action pair to probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes.
        gamma: Discount factor.
    
    Returns:
        A numpy array that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # returns_avg = defaultdict(float)
    # returns_count = defaultdict(float)
    # V = defaultdict(float)
    returns_count = np.zeros(SHAPE[:-1])
    V = np.zeros(SHAPE[:-1])
    
    for _ in range(num_episodes):
        episode = generate_episodes(policy, env)
        G = 0
        met_states = {}
        for state, action, reward in episode[::-1]:
            G = gamma*G + reward
            if state not in met_states:#first-visit MC prediction
                V[state] += (G-V[state])/(returns_count[state]+1)
                returns_count[state]+=1
                met_states[state] = False
    
    return V