#from agents.BaseAgent import BaseAgent
import numpy as np
import copy
import random

class Agent():
    
    def __init__(self, num_obs, num_messages, num_actions, epsilon, learning_rate, idx, T=1):
        
        self.num_obs = num_obs
        self.num_messages = num_messages
        self.num_actions = num_actions

        self.pi_comm = np.ones((self.num_obs, self.num_messages))
        self.pi_act = np.ones((self.num_messages, self.num_actions))  #num_actions = num_obs

        self.q_comm = np.ones((self.num_obs, self.num_messages))
        self.q_act = np.ones((self.num_messages, self.num_actions))  #num_actions = num_obs

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.idx = idx

        self.T = T

    def init_policies(self, x0_s, x0_r):
        for o in range(self.num_obs):
            self.pi_comm[o,:] = x0_s
        for o in range(self.num_messages):
            self.pi_act[o,:] = x0_r

    def define_data_savers(self, memory):
        self.pi_comm_t = np.ones((memory, self.num_obs, self.num_messages))
        self.pi_act_t = np.ones((memory, self.num_messages, self.num_actions))

    def save_data(self, i):
        self.pi_comm_t[i,:,:] = copy.deepcopy(self.pi_comm)
        self.pi_act_t[i,:,:] = copy.deepcopy(self.pi_act)

    def compute_entropy_comm(self):
        H = np.zeros((self.num_messages))
        
        for a in range(self.num_messages):
            for s in range(self.num_obs):
                H[a] -= self.pi_comm[s,a]*np.log(self.pi_comm[s,a])
        return H

    def compute_entropy_act(self):
        H = np.zeros((self.num_actions))
        
        for a in range(self.num_actions):
            for s in range(self.num_messages):
                H[a] -= self.pi_act[s,a]*np.log(self.pi_act[s,a])
        return H

    def argmax(self, values):

        top_value = float("-inf")
        ties = []

        for i in range(len(values)):
            if (values[i] > top_value):
                top_value = values[i]
                ties = []
            if (values[i] == top_value):
                ties.append(i)
        return np.random.choice(ties)
    
    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
        
    def epsilon_greedy_action(self, s, sender):

        if (sender == True):
            q = self.q_comm
            num_moves = self.num_messages
        else:
            q = self.q_act
            num_moves = self.num_actions
        
        if (random.uniform(0, 1) < self.epsilon):
            act = self.argmax(q[s])
        else:
            act = random.choice([i for i in range(num_moves)])

        return act

    def softmax_action(self, s, sender):

        if (sender == True):
            q = self.q_comm
            num_moves = self.num_messages
        else:
            q = self.q_act
            num_moves = self.num_actions

        c = np.max(q[s,:])
        #print("q[s,:]=", q[s,:])
        numerator = np.exp((q[s,:]-c)/self.T)
        #print("num=",numerator)
        denominator = np.sum(numerator)
        pi = numerator/denominator

        #denom = np.sum(np.exp(q[s,:]/self.T))
        #num = [np.exp(q[s,a]/self.T) for a in range(num_moves)]
        #pi = num/denom
        #print("pi=",pi)        
      
        if (sender == True):
            self.pi_comm[s,:] = pi
        else:
            self.pi_act[s,:] = pi

        act = np.random.choice([i for i in range(num_moves)], 1, p=pi)

        return act[0]
    
    def update_q_sender(self, rew, obs, mex):
        self.q_comm[obs, mex] += self.learning_rate*(rew - self.q_comm[obs, mex])
        
    def update_q_receiver(self, rew, mex, action):
        self.q_act[mex, action] += self.learning_rate*(rew - self.q_act[mex, action])
        
    def get_q_sender(self):
        return self.q_comm
    
    def get_q_receiver(self):
        return self.q_act

    def get_pi_sender(self):
        return self.pi_comm
    
    def get_pi_receiver(self):
        return self.pi_act