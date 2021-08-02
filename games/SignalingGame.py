import numpy as np
import random
import copy
from agents.Agent import Agent

class SignalingGame():
    
    def __init__(self, game, words_length, learning_rate, epsilon, epsilon_step=0.05, T = 1, dynamics=False, agents_fixed=False):
        self.num_agents = 2
        self.dict_agents = {}
        self.dynamics = dynamics
        self.agents_fixed = agents_fixed

        self.env = game

        self.T = T

        self.num_obs = self.env.num_obs
        self.num_messages = self.num_obs
        self.num_actions = self.num_obs

        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.learning_rate = learning_rate

        self.words_length = words_length
        
        assert(self.words_length >= self.num_obs)
        
        self.create_system()

    def init_policies(self, x0, y0):
        assert(np.sum(x0) == 1.)
        assert(np.sum(y0) == 1.)
        self.dict_agents[0].init_policy(x0)
        self.dict_agents[1].init_policy(y0)

    def create_system(self):
        for i in range(self.num_agents):
            self.dict_agents[i] = Agent(self.num_obs, self.num_messages, self.num_actions, self.epsilon, self.learning_rate, i, self.T)
            
    def train_step(self):

        if (self.agents_fixed == False):
            sender_idx, receiver_idx = random.sample([i for i in range(self.num_agents)], 2)
            self.sender = self.dict_agents[sender_idx]
            self.receiver = self.dict_agents[receiver_idx]
            #print("sender_idx=", self.sender.idx)

        obs = self.env.get_observation()
        
        #mex = self.sender.epsilon_greedy_action(obs, sender=True)
        #action = self.receiver.epsilon_greedy_action(mex, sender=False)

        mex = self.sender.softmax_action(obs, sender=True)
        action = self.receiver.softmax_action(mex, sender=False)

        rew_s, rew_r = self.env.get_reward(obs, action)

        self.sender.returns.append(rew_s)
        self.receiver.returns.append(rew_r)
        
        self.sender.update_q_sender(rew_s, obs, mex)
        self.receiver.update_q_receiver(rew_r, mex, action)

    def save_data(self, t):
        self.sender_states_t[:,:,int(t/self.save_step)] = self.sender.q_env
        self.receiver_states_t[:,:,int(t/self.save_step)] = self.receiver.q_act
        
    def train(self, num_loops):

        i = 0
        j = 0

        self.pi_sender_t = np.zeros((num_loops, self.num_obs, self.num_actions))
        self.pi_receiver_t = np.zeros((num_loops, self.num_obs, self.num_actions))

        self.pi_sender_t[0,:,:] = self.dict_agents[0].pi[:,:]
        self.pi_receiver_t[0,:,:] = self.dict_agents[1].pi[:,:]

        self.entropy_sender_t = np.zeros((num_loops, self.num_obs))
        self.entropy_receiver_t = np.zeros((num_loops, self.num_obs))

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
                print("i= ", i)

            for idx_ag in self.dict_agents:
                self.dict_agents[idx_ag].epsilon = self.epsilon
            
            self.train_step()

            if (self.dynamics and i%self.save_step==0):
                self.save_data(i)

            self.pi_sender_t[i,:,:] = copy.deepcopy(self.dict_agents[0].pi[:,:])
            self.pi_receiver_t[i,:,:] = copy.deepcopy(self.dict_agents[1].pi[:,:])                            
            send_ent = [self.sender.compute_entropy(i) for i in range(self.num_obs)]
            rec_ent = [self.receiver.compute_entropy(i) for i in range(self.num_obs)]
            
            self.entropy_sender_t[i, :] = send_ent
            self.entropy_receiver_t[i, :] = rec_ent
            
            i += 1

        sender_entropy = [self.sender.compute_entropy(i) for i in range(self.num_obs)]
        print("sender_entropy", sender_entropy)
        receiver_entropy = [self.receiver.compute_entropy(i) for i in range(self.num_obs)]
        print("receiver_entropy", receiver_entropy)

        print("final policy sender =\n", self.sender.pi)
        print("final policy receiver =\n", self.receiver.pi)
            
    def print_matrices(self, idx_agent):

        print("sender matrix:")
        self.sender.print_q_sender()       
        
        print("receiver matrix:")
        self.receiver.print_q_receiver()
        
    def return_agents_dict(self):
        return self.dict_agents