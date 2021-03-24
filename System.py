import numpy as np
from agents.Agent import DiscAgent
import random

class System:

    def __init__(self, num_agents, config, env):
        self.num_agents = num_agents
        self.config = config
        self.env = env
        self.adj_mat = None
        self.define_agents()
        self.returns = []

    def fill_adj_mat(self, adj_mat):
        self.adj_mat = adj_mat

    def define_agents(self):
        self.agents = {i:DiscAgent(self.config, self.env, i) for i in range(self.num_agents)}

    def return_dict_agents(self):
        return self.agents

    def get_reward(self, obs, act):
        if (obs == act):
            return 1
        return 0

    def choose(self, sender_idx):
        if (self.adj_mat):
            random.choice([self.agents[i] for i in range(self.num_agents) if self.adj_mat[sender_idx][i] == 1])
        else:
            return random.choice([self.agents[i] for i in range(self.num_agents) if i != sender_idx])

    def interaction_step(self):
        sender = random.choice(list(self.agents.values()))
        receiver = self.choose(sender.idx)

        obs = self.env.get_observation()
        
        mex = sender.generate_message(obs)
        act = receiver.perform_action(mex)
        
        rew = self.get_reward(obs, act)
        self.returns.append(rew)

        return (sender, receiver, obs, mex, act, rew)

    def update_q_matrices(self, args):
        sender, receiver, obs, mex, act, rew = args
        sender.update_q_sender(rew, obs, mex)
        receiver.update_q_receiver(rew, mex, act)        

    def learning_loop(self, train_episodes):
        i = 0
        while(i < train_episodes):
            args = self.interaction_step()
            self.update_q_matrices(args)

            i += 1

    def test_loop(self, evaluation_episodes):
        self.test_matrix = np.zeros((self.num_agents, self.num_agents))
        self.counter_matrix = np.zeros((self.num_agents, self.num_agents))
        
        i = 0
        while (i < evaluation_episodes):
            args = self.interaction_step()
            sender, receiver, _, _, _, rew = args
            self.test_matrix[sender.idx, receiver.idx] += rew
            self.counter_matrix[sender.idx, receiver.idx] += 1
            i +=1
            
        self.accuracy = np.true_divide(self.test_matrix, self.counter_matrix, where=(self.test_matrix!=0))
        
        print("test matrix:")
        print(self.test_matrix)
        
        print("counter:")
        print(self.counter_matrix)
        
        print("accuracy:")
        np.set_printoptions(precision=3)
        print(self.accuracy) 
