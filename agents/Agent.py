#from agents.BaseAgent import BaseAgent
import numpy as np
import random

class DiscAgent():
    
    def __init__(self, num_obs, num_actions, epsilon, lr, idx):
        self.epsilon = epsilon
        self.lr = lr
        self.idx = idx
        self.num_actions = num_actions
        self.num_messages = num_obs
        self.num_obs = num_obs
        self.q_env = np.ones((self.num_obs, self.num_messages))
        self.q_act = np.ones((self.num_messages, self.num_actions))
        
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
        
    def generate_message(self, obs):
        
        if (random.uniform(0, 1) < self.epsilon):
            return self.argmax(self.q_env[obs])
        else:
            return random.choice([i for i in range(self.num_messages)])
    
    def take_action(self, message):
        
        if (random.uniform(0, 1) < self.epsilon):
            return self.argmax(self.q_act[message])
        else:
            return random.choice([i for i in range(self.num_actions)])
    
    def update_q_sender(self, rew, obs, message):
        self.q_env[obs, message] += self.lr*(rew - self.q_env[obs, message])
        
        tmp_act = obs
        self.q_act[message, tmp_act] += self.lr*(rew - self.q_act[message, tmp_act])
        
    def update_q_receiver(self, rew, message, action):
        self.q_act[message, action] += self.lr*(rew - self.q_act[message, action])
        
        tmp_obs = action
        self.q_env[tmp_obs, message] += self.lr*(rew - self.q_env[tmp_obs, message])
        
    def _print(self):
        print("Hi! I am agent number", self.idx)
        
    def print_q_sender(self):   
        print(self.q_env)
    
    def print_q_receiver(self):
        print(self.q_act)
        
    def get_q_sender(self):
        return self.q_env
    
    def get_q_receiver(self):
        return self.q_act