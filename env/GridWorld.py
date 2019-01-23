import numpy as np
import io, sys
from gym.envs.toy_text import discrete

"""
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS

    T  1  2  3
    4  5  6  7
    8  9  a  b
    c  d  e  T

    """

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [4,4]):
        nS = np.prod(shape)
        nA = 4
        self.shape = shape

        grid = np.arange(nS).reshape(shape)
        it=np.nditer(grid, flags = ['multi_index'])

        P = {}#Probabitily
        MAX_Y = shape[0]
        MAX_X = shape[1]

        def is_terminal(s):
            return 1 if s==0 or s==(nS-1) else 0

        while not it.finished:
            s=it.iterindex
            y,x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a:[] for a in range(nA)}

            #is_done = lambda s:s==0 or s==(nS-1)
            is_done = is_terminal(s)
            reward = is_done-1.0

            #Terminal
            if is_done:
                for i in range(nA):
                    P[s][i] = [(1.0, s, reward, 1)]
            #Not terminal        
            else:
                ns = [s,s,s,s]
                ns[UP] = s if y == 0 else s - MAX_X
                ns[RIGHT] = s if x == (MAX_X - 1) else s + 1
                ns[DOWN] = s if y == (MAX_Y - 1) else s + MAX_X
                ns[LEFT] = s if x == 0 else s - 1
                for i in range(nA):
                    P[s][i] = [(1.0, ns[i], reward, is_terminal(ns[i]))]

            it.iternext()

        #Initial state distribution
        isd = np.ones(nS)/nS

        self.P = P

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)


    def _render(self,mode='human',close = False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            '''
            def reset(self):
                self.s = categorical_sample(self.isd, self.np_random)
                self.lastaction=None
                return self.s
            '''
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

