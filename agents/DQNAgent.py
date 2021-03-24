import torch as t
import numpy as np
from networks.DeepQNetwork import DeepQNetwork

class DQNAgent():

    def __init__(self, input_dim, num_actions, batch_size, lr, eps_start, intermed_nodes = 10, eps_min=0.01, eps_dec=5e-4, capacity=100000):
            
        self.lr = lr
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.action_space = [i for i in range(0, self.num_actions)]

        self.batch_size = batch_size
        self.capacity = capacity
        self.mem_counter = 0

        self.intermed_nodes = intermed_nodes
        self.Q = DeepQNetwork(self.lr, input_dim = input_dim, num_actions = num_actions, n_intermed_nodes = 3)
    
        self.state_memory = np.zeros((self.capacity, self.input_dim), dtype = np.float32)
        self.action_memory = np.zeros((self.capacity, self.input_dim), dtype = np.int64)
        self.reward_memory = np.zeros((self.capacity), dtype=np.float32)

        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec

    def store_transition(self, s, a, r):
        index = self.mem_counter % self.capacity
        self.state_memory[index] = s
        self.action_memory[index] = a
        self.reward_memory[index] = r
        
        self.mem_counter += 1

    def take_action(self, state):

        if (np.random.random() > self.epsilon):
            X = t.tensor(np.array([state]), device = self.Q.device, dtype=t.float)
            action = self.Q(X).argmax()
            return action
        else:
            return np.random.choice(self.action_space)

    def get_action_probabilities(self, state):
        X = t.tensor(np.array([state]), device = self.Q.device, dtype=t.float)
        return self.Q(X).detach().numpy()

    def train(self, idx):

        if (idx < self.batch_size):
            return None

        self.Q.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.capacity)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
                
        state_batch = t.tensor(self.state_memory[batch]).to(self.Q.device)
        reward_batch = t.tensor(self.reward_memory[batch]).to(self.Q.device)
        action_batch = t.tensor(self.action_memory[batch]).to(self.Q.device)

        Q = self.Q.forward(state_batch).to(self.Q.device).gather(1, action_batch)
        q_target = reward_batch.view(-1,1)

        loss = self.Q.loss(q_target, Q).to(self.Q.device)

        loss.backward()

        self.Q.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        