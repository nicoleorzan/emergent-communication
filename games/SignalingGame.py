import numpy as np
import random
from agents.Agent import DiscAgent
from utils.Environments import DiscEnv

class SignalingGame():
    
    def __init__(self, num_agents, num_obs, epsilon, learning_rate, words_length):
        self.num_agents = num_agents
        self.num_obs = num_obs
        self.num_actions = num_obs
        self.dict_agents = {}

        self.env = DiscEnv(self.num_obs, self.num_actions)

        self.epsilon = epsilon
        self.learning_rate = learning_rate

        self.words_length = words_length
        self.adj_matrix = None
        
        assert(self.words_length >= self.num_obs)
        
        self.create_system()
        
    def fill_adj_matrix(self, admat):
        self.adj_matrix = admat
        
    def create_system(self):
        for i in range(self.num_agents):
            self.dict_agents[i] = DiscAgent(self.epsilon, self.learning_rate, self.num_actions, self.num_obs, i)
            
    def get_reward(self, obs, action):
        if (obs == action):
            return 1
        return 0
            
    def perform_step(self):
        
        if (self.adj_matrix):
            sender_idx = random.choice([i for i in range(self.num_agents)])
            receiver_idx = random.choice([i for i in range(self.num_agents) if self.adj_matrix[sender_idx][i] == 1])
        else: 
            sender_idx, receiver_idx = random.sample([i for i in range(self.num_agents)], 2)
        
        sender = self.dict_agents[sender_idx]        
        receiver = self.dict_agents[receiver_idx]
        
        obs = self.env.get_observation()
        
        mex = sender.generate_message(obs)
        action = receiver.take_action(mex)
        
        rew = self.get_reward(obs, action)
        
        sender.update_q_sender(rew, obs, mex)
        receiver.update_q_receiver(rew, mex, action)
        
    def train(self, num_loops):
        
        i = 0
        
        frac = num_loops/10
        j = 0
        while (i < num_loops):
            # update epsilon
            if (i == frac*j):
                j += 1
                self.epsilon += 0.1
                if (self.epsilon > 1):
                    self.epsilon = 1
                print("epsilon = ", self.epsilon)
                for idx_ag in self.dict_agents:
                    self.dict_agents[idx_ag].epsilon = self.epsilon
            self.perform_step()
            
            i += 1
            
    def print_matrices(self, idx_agent):
        agent = self.dict_agents[idx_agent]
        
        print("idx agent=", idx_agent)
        print("sender matrix:")
        agent.print_q_sender()       
        
        print("receiver matrix:")
        agent.print_q_receiver()
        
    def returnum_agents_dict(self):
        return self.dict_agents
    
    def test_loop(self, num_loops):
        
        self.test_matrix = np.zeros((self.num_agents, self.num_agents))
        self.counter_matrix = np.zeros((self.num_agents, self.num_agents))
        
        i = 0
        while (i < num_loops):
            self.test()
            i +=1
            
        self.accuracy = self.test_matrix/self.counter_matrix
        
        print("test matrix:")
        print(self.test_matrix)
        
        print("counter:")
        print(self.counter_matrix)
        
        print("accuracy:")
        print(self.accuracy)
    
    def test(self):

        if (self.adj_matrix):
            sender_idx = random.choice([i for i in range(self.num_agents)])
            receiver_idx = random.choice([i for i in range(self.num_agents) if self.adj_matrix[sender_idx][i] == 1])
        else: 
            sender_idx, receiver_idx = random.sample([i for i in range(self.num_agents)], 2)
        
        sender = self.dict_agents[sender_idx]        
        
        receiver = self.dict_agents[receiver_idx]
        
        obs = self.env.get_observation()
        
        mex = sender.generate_message(obs)
        action = receiver.take_action(mex)
        
        rew = self.get_reward(obs, action)
        
        self.test_matrix[sender_idx, receiver_idx] += rew
        self.counter_matrix[sender_idx, receiver_idx] += 1