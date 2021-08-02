
import numpy as np
import random

def argmax(q_values):

    rand_generator = np.random.RandomState()

    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return rand_generator.choice(ties)


class Sender():

    def __init__(self, idx, lr, n_actions, epsilon):
        self.idx = idx
        self.lr = lr
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_prop = np.ones(self.n_actions)
        self.train = True

    def act(self, act_idx):
        return 1.0/self.n_actions*act_idx

    def epsilon_greedy_proposal(self):
        if (self.train == False):
            return argmax(self.q_prop)

        if (np.random.random() > self.epsilon):
            return argmax(self.q_prop)
        else:
            return random.randint(0, self.n_actions-1)

    def softmax_proposal(self):
        if (self.train == False):
            return argmax(self.q_prop)

        return np.random.choice([i for i in range(self.n_actions)], 1, p=self.softmax())[0]

    def softmax(self):
        denom = np.sum(np.exp(self.q_prop))
        self.p = np.exp(self.q_prop)/denom
        
    def TD_prop(self, act, rew):
        self.q_prop[act] += self.lr*(rew - self.q_prop[act])


class Receiver():

    def __init__(self, idx, lr, threshold, n_states, epsilon):
        self.idx = idx
        self.lr = lr
        self.threshold = threshold
        self.epsilon = epsilon
        self.n_actions = 2 # 0 no, 1 yes
        self.n_states = n_states
        self.q_recv = np.ones((self.n_states, self.n_actions))
        self.train = True

    def softmax(self):
        denom = np.sum(np.exp(self.q_recv))
        self.p = np.exp(self.q_recv)/denom

    def epsilon_greedy_recv(self, p):
        if (p < self.threshold):
            return 0 # rifiuto sempre se minore della soglia

        if (self.train == False):
            return argmax(self.q_recv[p,:])

        if (np.random.random() > self.epsilon):
            return argmax(self.q_recv[p,:])
        else:
            return random.randint(0, self.n_actions-1)

    def softmax_recv(self, p):
        if (p < self.threshold):
            return 0 # rifiuto sempre se minore della soglia

        if (self.train == False):
            return argmax(self.q_recv[p,:])

        return np.random.choice([i for i in range(self.n_actions)], 1, p=self.softmax())[0]
           
    def threshold_recv(self, p):
        if (p >= self.threshold):
            return 1 # accetto
        else:
            return 0 # rifiuto

    def TD_recv(self, state, act, rew):
        self.q_recv[state,act] += self.lr*(rew - self.q_recv[state,act])