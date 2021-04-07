import torch as t
import numpy as np
from networks.DRQN import DRQN
from utils.ExperienceReplay import ExperienceReplay

class DRQNAgent():

    def __init__(self, input_dim, num_actions, batch_size, lr, eps_start, intermed_nodes = 10, eps_min=0.01, eps_dec=5e-4, capacity=100000):
            
        self.lr = lr
        self.gamma = 0.99
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.action_space = [i for i in range(0, self.num_actions)]

        self.batch_size = batch_size
        self.capacity = capacity
        self.mem_counter = 0

        self.intermed_nodes = intermed_nodes
        self.Q = DRQN(self.lr, input_dim = input_dim, num_actions = num_actions, n_layers = 2)
        self.target_Q = DRQN(self.lr, input_dim = input_dim, num_actions = num_actions, n_layers = 2)
    
        self.memory = ExperienceReplay()

        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.target_net_update_freq = 100
        self.update_count = 0

    def store_transition(self, s, a, r, _s):
        self.memory.push((s, a, r, _s))

    def prep_minibatch(self):
        transitions = self.memory.sample(self.batch_size)

        state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)
        
        #shape = (-1,)#+self.input_dim
        #print("shape=", shape)
        state_batch = t.tensor(state_batch, device = self.Q.device, dtype = t.float).view(-1,1)#.view(shape)        
        #print("state batch=", state_batch)
        action_batch = t.tensor(action_batch, device = self.Q.device, dtype = t.long).squeeze().view(-1, 1)
        reward_batch = t.tensor(reward_batch, device = self.Q.device, dtype = t.float).squeeze().view(-1, 1)
        #print("action batch=", action_batch)
        #print("reward batch=", reward_batch)

        #print("next state batch=", next_state_batch)

        non_final_mask = t.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=self.Q.device, dtype=t.long)
        #print("non final mask=", non_final_mask)
        try: #sometimes all next states are false
            non_final_next_states = t.tensor([s for s in next_state_batch if s is not None], device=self.Q.device, dtype=t.float).view(-1,1)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        #next_state_batch = t.tensor(next_state_batch, device = self.Q.device, dtype = t.float)        
        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, empty_next_state_values

        #return state_batch, action_batch, reward_batch, next_state_batch

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

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_Q.load_state_dict(self.Q.state_dict())

    def get_max_next_state_action(self, next_states):
        #print("next =", next_states)
        #print("target q(next)=", self.target_Q(next_states))
        return self.target_Q(next_states).max(dim=1)[1]#.view(-1, 1)

    def train(self, idx):

        #self.store_transition(s, a, r, _s) si fa esternamente

        if (idx < self.batch_size):
            return None

        self.Q.optimizer.zero_grad()        

        batch_vars = self.prep_minibatch()

        # compute loss

        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars
    
        current_q_values = self.Q(batch_state).gather(1, batch_action)
        #print("current q=", current_q_values)

        #print("non final next s", non_final_next_states)

        with t.no_grad():
            max_next_q_values = t.zeros(self.batch_size, device=self.Q.device, dtype=t.float).unsqueeze(dim=1)
            if not empty_next_state_values:
               max_next_action = self.get_max_next_state_action(non_final_next_states).view(-1, 1)
               #print("max next action=", max_next_action)
               max_next_q_values[non_final_mask] = self.target_Q(non_final_next_states).gather(1, max_next_action)
               #print("max next q values=", max_next_q_values)
               #print("batch_reward=", batch_reward)
            expected_q_values = batch_reward + self.gamma*max_next_q_values
        #print("expected q=", expected_q_values)

        #print(current_q_values.shape)
        #print(expected_q_values.shape)
        diff = (expected_q_values - current_q_values)

        loss = self.Q.loss(expected_q_values, current_q_values).to(self.Q.device)
        loss.backward()

        self.Q.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        