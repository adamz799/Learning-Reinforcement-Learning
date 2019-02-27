import gym
import numpy as np
import itertools
import sys
import collections
import math

import torch
from torch import nn, optim
import torch.nn.init as init
from tensorboardX import SummaryWriter

if "../" not in sys.path:
  sys.path.append("../") 

from env.CliffWalking import CliffWalkingEnv

device = torch.device('cpu')
writer = SummaryWriter()
env = CliffWalkingEnv()

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n, bias=True),
            nn.Softmax()
        )
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight, std=0.01)
                init.normal_(layer.bias, std=0.01)
        #torch.autograd.set_grad_enabled(True)

    def forward(self, inputs):
        return self.net(inputs)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True),
        )
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight, std=0.01)
                init.normal_(layer.bias, std=0.01)
        #torch.autograd.set_grad_enabled(True)

    def forward(self, inputs):
        return self.net(inputs)
     

def act(state, policy):     
    P = policy(state).detach().cpu().numpy()
    return np.random.choice(np.arange(env.nA), p=P)

def generate_episodes(policy, env):
    """
    step = (torch Tensor of state, action, reward)
    """
    episode = []
    done = False
    state = env.init_state
    while not done:
        state1 = torch.Tensor(state)
        action = act(state1, policy)
        next_state, reward, done = env.step(state, action)
        episode.append((state1, action, reward))
        state = next_state
    
    return episode
    
def policy_loss_func(state, action, policy):
    P = policy(state)[action]
    return torch.log(P)


def reinforce(gamma=0.95, alpha_w = 1e-4, alpha_theta = 1e-4, num_eposide = 1000):
    policy = PolicyNet()
    value = ValueNet()
    i =0
    value_loss_func = nn.L1Loss()
    value_optimizer = optim.SGD(value.parameters(), lr = alpha_w)
    policy_optimizer = optim.SGD(policy.parameters(), lr = alpha_theta)

    #policy_loss_func = nn.
    for j in range(num_eposide):
        G = torch.Tensor([0.0])
        episode = generate_episodes(policy, env)
        t = len(episode) - 1
        g = math.pow(gamma,t)
        for state, action, reward in episode[::-1]:
            G = gamma*G + reward
            # Update w
            value_loss = value_loss_func(G,value(state))
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            # Update theta
            policy_loss = policy_loss_func(state, action, policy)*value_loss.detach()*g
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            g /= gamma
            i+=1
            writer.add_scalar('data/value_loss', value_loss, i)
            writer.add_scalar('data/policy_loss', policy_loss, i)

        print('Episode {0} is visited'.format(j))

    torch.save(policy, 'policy_net.pkl')
    torch.save(value, 'value_net.pkl')

    return policy, value


if __name__ == "__main__":
    reinforce()
    writer.export_scalars_to_json("Double_DQN_test.json")
    writer.close()

        
