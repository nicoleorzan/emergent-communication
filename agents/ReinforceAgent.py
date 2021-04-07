import numpy as np
import random

class ReinforceAgent():

    def __init__(self, num_actions, idx):
        self.idx = idx
        self.num_actions = num_actions

    def policy(self, state, w):
        z = state.dot(w)
        exp = np.exp(z)
        return exp/np.sum(exp)

    def softmax_grad(self, softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def take_action(self, s, w):
        p = self.policy(s, w)
        action = np.random.choice(self.num_actions,p=p[0])
        return action