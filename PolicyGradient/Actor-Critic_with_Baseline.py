import numpy as np
import math, sys

import torch
from torch import nn, optim
import torch.nn.init as init
from tensorboardX import SummaryWriter

if "../" not in sys.path:
  sys.path.append("../") 

from env.CliffWalking import CliffWalkingEnv
from REINFORCE_with_Baseline import PolicyNet, ValueNet

device = torch.device('cpu')
writer_AC = SummaryWriter()
env = CliffWalkingEnv()


def policy_loss_func(state, action, policy):
    P = policy(state)[action]
    return torch.log(P)


def actor_critic(gamma=0.95, alpha_w = 1e-4, alpha_theta = 1e-4, num_eposide = 1000):
    policy = PolicyNet()
    value = ValueNet()
    i =0
    value_loss_func = nn.L1Loss()
    value_optimizer = optim.SGD(value.parameters(), lr = alpha_w)
    policy_optimizer = optim.SGD(policy.parameters(), lr = alpha_theta)

    for j in range(num_eposide):
        I = torch.Tensor([1.0])
        state = torch.Tensor(env.init_state)
        while 1:
            P = policy(state).detach().cpu().numpy()
            action = np.random.choice(np.arange(env.nA), p=P)
            next_state, reward, is_done = env.step(state, action)
            next_state = torch.Tensor(next_state)
            
            if is_done:
                delta = reward -value(state)
            else:
                delta = reward+gamma*value(next_state).detach() - value(state)
            
            # Update w
            value_loss = delta
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            # Update theta
            policy_loss = policy_loss_func(state, action, policy)*delta.detach()*I
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            if is_done:
                break
            else:
                I *= gamma
                state = next_state
                i+=1
                writer_AC.add_scalar('data/value_loss', value_loss, i)
                writer_AC.add_scalar('data/policy_loss', policy_loss, i)

        print('Episode {0}'.format(j))

    torch.save(policy, 'AC_policy_net.pkl')
    torch.save(value, 'AC_value_net.pkl')

    return policy, value


if __name__ == "__main__":
    actor_critic()
    writer_AC.export_scalars_to_json("Double_DQN_test.json")
    writer_AC.close()

        
