# Windy Gridworld

import numpy as np
import io, sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridWorldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [7,10]):
        nA = 4
        nS = np.prod(shape)  
        delta = np.array([[-1,0],[0,1],[1,0],[0,-1]])
        self.shape = shape
        self.termination = (3,7)
        self.init_state = (3,0)

        grid = np.arange(nS).reshape(shape)
        it=np.nditer(grid, flags = ['multi_index'])
        
        # Wind strength
        winds = np.zeros(shape, int)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        P = {}#Probabitily
        
        while not it.finished:
            #s = it.iterindex
            y,x = it.multi_index
            s = (y,x) # Use tuple of index to represent state
            P[s] = {}
            # P[s][a] = (prob, next_state, reward, is_done)
            for a in range(nA):
                P[s][a] = self._calculate_transition_prob(s, delta[a], winds)

            it.iternext()

        # Initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        self.P = P

        super(WindyGridWorldEnv, self).__init__(nS, nA, P, isd)

    
    def _render(self,mode='human',close = False):
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

    def _is_terminal(self, s):
            return 1 if s==self.termination else 0

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, s, delta, winds):
        """
        Args:
            s: Tuple(y,x), current state.
            delta: Numpy array, displacement cause by action.
            winds: Numpy array, wind strength.

        Return:
            Tuple(prob, next_state, reward, is_done)
        """
        new_s = np.array(s) + delta + np.array([-1, 0]) * winds[s]
        new_s = tuple(self._limit_coordinates(new_s).astype(int))
        is_done = self._is_terminal(s)
        reward = is_done - 1.0
        return (1.0, new_s, reward, is_done)


    def step(self, state, action):
        _, next_state, reward, is_done = self.P[state][action]
        return (next_state, reward, is_done)

        # This part should be re-implemented if the transition of env is different
        # For example:
        #
        # trans_prob = [self.P[state][a][0] for a in self.nA]
        # next_action = np.random.choice(np.arange(self.nA),p=trans_prob)
        # _, next_state, reward, is_done = self.P[state][next_action]
        # return ...
