import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from GridWorld import GridWorldEnv

env = GridWorldEnv()

def policy_evaluation(policy, env, gamma = 0.95, theta = 1e-5):
    '''
    The environment’s dynamics are completely known.
    
    V(s) = ∑π(a|s)∑P(n_s,r|s,a)[r+γ*V(n_s)]

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        gamma: Discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    '''
    
    V = np.zeros(env.nS)#value
    iteration = 1
    while 1:
        flag=1
        #update
        for s in range(env.nS-1, -1, -1):
            v1 = 0
            for a in range(env.nA):
                #V[s] = policy[s,a]*env.P[s][a]
                r1=0
                for t in env.P[s][a]:
                    # P[s][a] = (prob, next_state, reward, is_done)
                    # Sadly P is not a numpy array
                    r1 += t[0]*(t[2] + gamma*V[1])
                v1 += policy[s,a]*r1

            flag_t = 1 if abs(V[s]-v1)<theta else 0
            flag *= flag_t
            V[s] = v1
        print('iteration: ', iteration)
        iteration+=1
        #Check whether delta<theta
        if flag or iteration>10000:
            break

    return np.array(V)

random_policy = np.ones([env.nS, env.nA])/env.nA
