import random
import numpy as np
import cv2
import sys
sys.path.append("game/")
from tensorboardX import SummaryWriter
import wrapped_flappy_bird as game

import torch
from torch import nn, optim
import torch.nn.init as init
import torch.multiprocessing as mp
from Net import Net

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
BETA = 0.01
T_MAX=1000

device = torch.device('cpu')
writer = SummaryWriter()

def train(net, t, T_counter, optimizer, loss_func):
    game_state = game.GameState()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # input_actions[0] == 1: do nothing
    # input_actions[1] == 1: flap the bird
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    init_state = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, init_state = cv2.threshold(init_state, 1, 255, cv2.THRESH_BINARY)
    init_state = init_state.astype(np.float32)[np.newaxis,np.newaxis,:,:]  
    state = torch.from_numpy(init_state).float().to(device)

    while 1:

        t_start=t
        episode = []
        optimizer.zero_grad()
        while 1:
            prob, value = net(state)
            action = np.zeros(ACTIONS)
            action_index = prob.sample()
            action[action_index]=1

            next_state, reward, is_done = game_state.frame_step(action)
            next_state = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
            _, next_state = cv2.threshold(next_state, 1, 255, cv2.THRESH_BINARY)
            next_state = next_state.astype(np.float32)[np.newaxis,np.newaxis,:,:]   
            next_state = torch.from_numpy(next_state).float().to(device)

            t+=1
            T_counter.value += 1
            print('t: {0}, T: {1}'.format(t, T_counter.value))

            if is_done :
                R = 0
                break
            elif t-t_start==5:
                R = value.item()
                break
            else:
                episode.append((value, action_index, reward, prob))
                state = next_state
                

        for value, action_index, reward, prob in episode[::-1]:
            R = reward + GAMMA*R
        loss = -prob.log_prob(action_index)* (R - value.detach()) - BETA*prob.entropy()
        loss_v = loss_func(R, value)
        loss.backward(retain_graph=True)
        loss_v.backward()

        optimizer.step()

        if T_counter.value > T_MAX:
            return


if __name__ == "__main__":
    num_process = 2
    net = Net().to(device)
    net.share_memory()
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.99)
    loss_func = nn.MSELoss().to(device)
    t=0
    T_counter = mp.Value('l',0)
    process = []
    
    for rank in range(num_process):
        p = mp.Process(target = train, args = (net, t, T_counter, optimizer, loss_func, ))
        p.start()
        process.append(p)
    for p in process:
        p.join()

    torch.save(net, 'A3C.pkl')
  
