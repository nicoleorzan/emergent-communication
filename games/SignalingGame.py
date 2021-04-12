import numpy as np
import random
from agents.Agent import DiscAgent
from utils.Environments import RockPaperScissor, CoordinationGame

class SignalingGame():
    
    def __init__(self, words_length, learning_rate, epsilon, epsilon_step=0.05, dynamics=False, agents_fixed=False, game_cooperative=False):
        self.num_agents = 2
        self.dict_agents = {}
        self.dynamics = dynamics
        self.agents_fixed = agents_fixed
        self.game_cooperative = game_cooperative

        if (self.game_cooperative):
            self.env = CoordinationGame()
        else:
            self.env = RockPaperScissor()

        self.num_obs = self.env.num_obs
        self.num_actions = self.num_obs

        #assert(self.num_obs == 3)
        #assert(self.num_actions == 3)

        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.learning_rate = learning_rate

        self.words_length = words_length
        
        assert(self.words_length >= self.num_obs)
        
        self.create_system()

    def create_system(self):
        for i in range(self.num_actions):
            self.dict_agents[i] = DiscAgent(self.num_obs, self.num_actions, self.epsilon, self.learning_rate, 0)
            
    def train_step(self):

        if (self.agents_fixed == False):
            sender_idx, receiver_idx = random.sample([i for i in range(self.num_agents)], 2)
            self.sender = self.dict_agents[sender_idx]
            self.receiver = self.dict_agents[receiver_idx]
            #print("sender_idx=", self.sender.idx)

        obs = self.env.get_observation()
        
        mex = self.sender.generate_message(obs)
        action = self.receiver.take_action(mex)
        
        rew_s, rew_r = self.env.get_reward(obs, action)
        
        self.sender.update_q_sender(rew_s, obs, mex)
        self.receiver.update_q_receiver(rew_r, mex, action)

    def update_data(self, t):
        self.sender_states_t[:,:,int(t/self.save_step)] = self.sender.q_env
        self.receiver_states_t[:,:,int(t/self.save_step)] = self.receiver.q_act
        
    def train(self, num_loops):

        i = 0
        j = 0

        if (self.agents_fixed):
            self.sender = self.dict_agents[0]
            self.receiver = self.dict_agents[1]

        if (self.dynamics == True):
            self.save_step = 10
            self.sender_states_t = np.zeros((self.num_obs, self.num_actions, int(num_loops/self.save_step)))
            self.receiver_states_t = np.zeros((self.num_obs, self.num_actions, int(num_loops/self.save_step)))
        
        frac = num_loops/10
        while (i < num_loops):

            # update epsilon
            if (i == frac*j):
                j += 1
                self.epsilon += self.epsilon_step
                if (self.epsilon > 1):
                    self.epsilon = 1
                print("epsilon = ", self.epsilon)
                for idx_ag in self.dict_agents:
                    self.dict_agents[idx_ag].epsilon = self.epsilon
            
            # learning step
            self.train_step()

            if (self.dynamics and i%self.save_step==0):
                self.update_data(i)
            
            i += 1
            
    def print_matrices(self, idx_agent):

        print("sender matrix:")
        self.sender.print_q_sender()       
        
        print("receiver matrix:")
        self.receiver.print_q_receiver()
        
    def return_agents_dict(self):
        return self.dict_agents