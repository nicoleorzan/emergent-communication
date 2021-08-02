import numpy as np
import random
import copy
from agents.Agent import Agent

class PopSignalingGame():
    
    def __init__(self, game, num_agents, num_obs, num_messages, num_actions, epsilon, epsilon_step, learning_rate, T):
        
        self.env = game

        self.num_agents = num_agents
        self.num_obs = num_obs
        self.num_messages = num_messages
        self.num_actions = num_actions

        self.dict_agents = {}
        self.env = game
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.learning_rate = learning_rate

        self.T = T
        
        self.adj_matrix = np.zeros((self.num_agents, self.num_agents))
                
        self.create_system()

    def init_policy(self, x0_s, x0_r):
        for i in range(self.num_agents):
            self.dict_agents[i].init_policies(x0_s, x0_r)
        
    def give_adj_matrix(self, adjmat):
        self.adj_matrix = adjmat

        print(self.adj_matrix)
        
    def create_system(self):

        for i in range(self.num_agents):
            self.dict_agents[i] = Agent(self.num_obs, self.num_messages, self.num_actions, self.epsilon, self.learning_rate, i, self.T)
            
    def perform_step(self):
        
        sender_idx = random.choice([i for i in range(self.num_agents)])

        li = [i for i in range(self.num_agents) if self.adj_matrix[sender_idx][i] == 1]
        if (li):
            receiver_idx = random.choice([i for i in range(self.num_agents) if self.adj_matrix[sender_idx][i] == 1])
        else:
            return
            
        sender = self.dict_agents[sender_idx]        
        receiver = self.dict_agents[receiver_idx]
        
        obs = self.env.get_observation()
        
        mex = sender.softmax_action(obs, sender=True)
        action = receiver.softmax_action(mex, sender=False)
        #mex = sender.epsilon_greedy_action(obs, sender=True)
        #action = receiver.epsilon_greedy_action(mex, sender=False)
        
        rew_s, rew_r = self.env.get_reward(obs, action)
        
        sender.update_q_sender(rew_s, obs, mex)
        receiver.update_q_receiver(rew_r, mex, action)
        
    def perform_loop(self, num_loops, eps_step = 10):
        
        i = 0
        
        self.eps_step = eps_step
        frac = num_loops/self.eps_step
        j = 0

        for idx_ag in self.dict_agents:
            self.dict_agents[idx_ag].define_data_savers(num_loops)

        while (i < num_loops):

            if (i == frac*j):
                j += 1
                self.epsilon += self.epsilon_step
                if (self.epsilon > 1):
                    self.epsilon = 1
                print("epsilon = ", self.epsilon)
                for idx_ag in self.dict_agents:
                    self.dict_agents[idx_ag].update_epsilon(self.epsilon)
            for idx_ag in self.dict_agents:
                self.dict_agents[idx_ag].save_data(i)
            self.perform_step()
            
            i += 1

        #for i in range(self.num_agents):
        #    self.dict_agents[i].compute_emtropy()
        
    def return_agents_dict(self):
        return self.dict_agents