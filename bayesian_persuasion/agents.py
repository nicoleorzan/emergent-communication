import random
from numpy.random import sample
from scipy.stats import bernoulli
from scipy.stats import rv_discrete 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions import Normal

verbose = False
# 0 = innocente, 1 = colpevole 

class Net(nn.Module):

    # funzioni attivazione, loss

    def __init__(self, input_size, hidden_size, n_actions, lr):
        super(Net, self).__init__()
                
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions #n_actions # 2: say i, say c
        self.output_size = 2

        self.policy = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.LogSoftmax(dim=1)
        )
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.training = True

    def forward(self, truth):

        out = self.policy(truth)

        dist = Categorical(logits=out)

        if self.training:
            sampled = dist.sample()
        else: 
            sampled = out.argmax(dim=1)

        logprobs = dist.log_prob(sampled)
        
        return dist.probs, sampled, logprobs



class Sender():

    def __init__(self, input_size, prior_i, lr = 0.1, beta=False):

        self.input_size = input_size
        self.prior_i = prior_i
        self.n_actions = 1 # number of samples out of the probability distribution
        self.actions_space = [i for i in range(0, 2)] # actions: [0, 1]
        self.p_cond = torch.zeros((2,2))
    
        self.model = Net(input_size=self.input_size, hidden_size=32, n_actions=self.n_actions, lr=lr)
      
        self.lr = lr
        self.baseline = 0.
        self.n_update = 0.
        self.ent_reg = 0
        self.training = True
   
    def loss(self, rew, logprobs):

        #loss = ( -rew.squeeze().detach() * logprobs ).mean()
        #print("rew=", rew)
        #print("logrpobs=", logprobs)
        #print(" -rew * logprobs =", -rew * logprobs )
        loss = ( -rew * logprobs ).mean()

        return loss


class DiscreteSender():

    def __init__(self, params):
        
        self.n_probs = params['num_probs']
        self.epsilon = params['epsilon0']
        self.eps_min = params['eps_min']
        self.eps_decay = params['eps_decay']
        self.lr = params['lr0']
        self.lr_decay = params['lr_decay']
        self.T = params['T']

        self.q = np.full((self.n_probs+1, self.n_probs+1), 2.) # le q rappresentano le probabilita di innocente per le due verita
        self.q_counts = np.zeros((self.n_probs+1, self.n_probs+1))
        self.rewards = np.zeros((params['num_probs']+1, params['num_probs']+1))
        self.p = np.ones((self.n_probs+1))
        self.p_cond = np.zeros((2,2))

        self.init_p()

    def init_p(self):
        for i in range(self.n_probs+1):
            self.p[i] = i/self.n_probs 
        self.p[self.p == 0.] = 0.000001
        self.p[self.p == 1.] = 1.-0.000001

    def argmax(self, q_values):

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

    def choose_signal(self):
        if (np.random.random() < self.epsilon):
            p_i_I_idx = random.randint(0, self.n_probs)
            p_i_C_idx = random.randint(0, self.n_probs)
        else:
            p_i_I_idx, p_i_C_idx = np.unravel_index(np.argmax(self.q, axis=None), self.q.shape) # row, column

        self.update_p_cond(p_i_I_idx, p_i_C_idx)

        self.q_counts[p_i_I_idx, p_i_C_idx] += 1

        return p_i_I_idx, p_i_C_idx
            

    def choose_signal_softmax(self):
        denom = np.sum(np.exp(self.q)/self.T)
        #print("denom=", denom)
        #print("np.exp(self.q)=", np.exp(self.q))
        probs = np.exp(self.q/self.T)/denom
        p_i_I_idx, p_i_C_idx = np.unravel_index(np.argmax(probs, axis=None), probs.shape) 

        self.update_p_cond(p_i_I_idx, p_i_C_idx)

        self.q_counts[p_i_I_idx, p_i_C_idx] += 1

        return p_i_I_idx, p_i_C_idx

    def act(self, act):
        return self.p[:,act]

    def update_p_cond(self, p_i_I_idx, p_i_C_idx):
        self.p_cond[0,0] = self.p[p_i_I_idx]
        self.p_cond[1,0] = 1 - self.p_cond[0,0]
        self.p_cond[0,1] = self.p[p_i_C_idx]
        self.p_cond[1,1] = 1 - self.p_cond[0,1] 

    def update(self, p_i_I_idx, p_i_C_idx, rew):
        if (self.q_counts[p_i_I_idx, p_i_C_idx] != 0):
            lr = 1./self.q_counts[p_i_I_idx, p_i_C_idx]
        else: 
            lr = 0.9
        self.q[p_i_I_idx, p_i_C_idx] += lr*(rew - self.q[p_i_I_idx, p_i_C_idx])

    def sample_mex(self, idx_truth):
        return np.random.choice([0,1], 1, p=self.p_cond[:, idx_truth])[0]


class Nature():

    def __init__(self, p0):
        self.p0 = p0
        self.action_space = [i for i in range(0, 2)] 
        self.p0 = 0.5

    def get_state_of_world(self, batch_size):

        idx_truth = torch.Tensor([np.random.choice(self.action_space, batch_size, p=np.array([self.p0, 1-self.p0]))]).to(torch.int64).squeeze()
        #truth = torch.reshape(idx_truth, (batch_size, idx_truth.shape[0]))
        truth = F.one_hot(idx_truth.unsqueeze(dim=0), num_classes=2)
        return truth, idx_truth

    def get_reward(self, real, predicted_r):

        send_rew = torch.zeros(real.shape)

        if (hasattr(real, '__len__') == 1):

            if (predicted_r == 1):
                return 1.
            elif (predicted_r == 0):
                return -1.

        for idx, _ in enumerate(real):
            if (predicted_r[idx] == 1):
                send_rew[idx] = 1.
            elif (predicted_r[idx] == 0):
                send_rew[idx] = -1.

        return send_rew

    def get_reward_fair(self, real, predicted_r):

        send_rew = torch.zeros(real.shape)
        #print("real=", real)
        #print("predicted=", predicted_r)

        if (hasattr(real, '__len__') == 0):
            if (real == predicted_r):
                return 1.
            else: 
                return -1.

        for idx, _ in enumerate(real):
        
            if (real[idx] == predicted_r[idx]):
                send_rew[idx] = 1.
            else: 
                send_rew[idx] = -1.

        return send_rew

    
    def get_reward_fair_discrete(self, real, predicted_r):

        send_rew = torch.zeros(real.shape)
        #print("real=", real)
        #print("predicted=", predicted_r)

        if (hasattr(real, '__len__') == 1):
            if (real == predicted_r):
                return 1.
            else: 
                return -1.

        for idx, _ in enumerate(real):
        
            if (real[idx] == predicted_r[idx]):
                send_rew[idx] = 1.
            else: 
                send_rew[idx] = -1.

        return send_rew



class Rational_Receiver():

    def __init__(self, prior_i):

        self.prior_i = prior_i # probab che sia innocente

    def predict(self, messages, p_conditioned_s):

        responses = torch.empty(messages.shape)
        p_i = p_conditioned_s[0,0]*self.prior_i + p_conditioned_s[0,1]*(1-self.prior_i)

        p_I_i = p_conditioned_s[0,0]*self.prior_i/p_i
        p_C_i = p_conditioned_s[0,1]*(1-self.prior_i)/p_i

        p_c = p_conditioned_s[1,0]*self.prior_i + p_conditioned_s[1,1]*(1-self.prior_i)

        p_C_c = p_conditioned_s[1,1]*(1-self.prior_i)/p_c
        p_I_c = p_conditioned_s[1,0]*self.prior_i/p_c

        #print("p_I_i=", p_I_i)
        #print("p_I_c=", p_I_c)
        #print("p_C_i=", p_C_i)
        #print("p_C_c=", p_C_c)

        if (p_i == 0):
            p_I_i = 0
            p_C_i = 0
        if (hasattr(messages, '__len__') == 0):

            if (messages == 0):
                #resp = np.random.choice([0,1], 1, p=[p_I_i, 1-p_I_i])[0]
                if (p_I_i > 0.5): 
                    resp = 0
                else: 
                    resp = 1

            else: 
                #resp = np.random.choice([0,1], 1, p=[p_I_c, 1-p_I_c])[0]
                if (p_C_c > 0.5): 
                    resp = 1
                else: 
                    resp = 0

            return resp, p_I_i, p_I_c, p_C_i, p_C_c

        for idx, mex in enumerate(messages):

            mex = int(mex.detach().numpy())

            if (mex == 0):

                #responses[idx] = np.random.choice([0,1], 1, p=[p_I_i, 1-p_I_i])[0]

                if (p_I_i > 0.5): 
                    responses[idx] = 0
                else: 
                    responses[idx] = 1

            elif (mex == 1):

                #responses[idx] = np.random.choice([0,1], 1, p=[p_I_c, 1-p_I_c])[0]
                if (p_C_c > 0.5):
                    responses[idx] = 1
                else: 
                    responses[idx] = 0
            
            else:
                print("Error!!")

        return responses, p_I_i, p_I_c, p_C_i, p_C_c