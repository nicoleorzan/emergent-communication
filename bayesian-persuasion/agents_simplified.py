
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm
import torch
from torch.distributions import Normal, Beta
from TruncatedNormal import TruncatedNormal

torch.manual_seed(0)

class MiniBeta(nn.Module):

    def __init__(self, input_size, hidden_size, n_actions, lr, drop_rate, min_var=1e-2):
        super(MiniBeta, self).__init__()
                
        self.dim = 1
        self.input_size = input_size # dim=1
        self.hidden_size = 4 # hidden_size
        self.n_actions = n_actions #n_actions # 2: say i, say c

        self.policy = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        self.alpha = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.dim),
            nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.dim),
            nn.Sigmoid()
        )

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(p=self.drop_rate)

        self.min_var = min_var
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.training = True

    def forward(self, truth):

        out = self.policy(truth)
        out_drop = self.drop(out)

        alpha = self.alpha(out_drop)
        beta = self.beta(out_drop) #+ self.min_var

        mean = alpha/(alpha+beta)
        var = alpha*beta/((alpha+beta)*(alpha+beta)*(alpha+beta+1))

        dist = Beta(alpha, beta) 
        
        if self.training:
            sampled = dist.rsample()
        else:
            sampled = mean
        
        logprobs = dist.log_prob(sampled)
        
        return  mean, var, sampled, logprobs



class MiniGaus(nn.Module):

    def __init__(self, input_size, hidden_size, n_actions, lr, drop_rate, min_var=1e-2):
        super(MiniGaus, self).__init__()
                
        self.dim = 1
        self.input_size = input_size # dim=1
        self.hidden_size = 4 # hidden_size
        self.n_actions = n_actions #n_actions # 2: say i, say c

        self.policy = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        self.mean = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.dim),
            nn.Sigmoid()
        )
        self.var = nn.Sequential(
            nn.Linear(self.hidden_size, self.dim),
            nn.Sigmoid()
        )

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(p=self.drop_rate)

        self.min_var = min_var
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.training = True

    def forward(self, truth):

        out = self.policy(truth)
        out_drop = self.drop(out)

        mean = self.mean(out_drop)
        var = self.var(out_drop) + self.min_var

        #dist = Normal(mean, var) 

        dist = TruncatedNormal(loc=mean, scale=var, a=0, b=1) 
        
        if self.training:
            sampled = dist.rsample()
        else:
            sampled = mean
        
        logprobs = dist.log_prob(sampled)
        
        return  mean, var, sampled, logprobs

class Sender_Mini():

    def __init__(self, prior_i, lr = 0.1, drop_rate = 0.2, beta=False):

        self.prior_i = prior_i
        self.input_size = 2
        self.n_actions = 1 # number of samples out of the probability distribution
        self.actions_space = [i for i in range(0, 2)] # [0,1]
        self.p_cond = np.zeros((2,2))
    
        self.model = MiniGaus(input_size=self.input_size, hidden_size=32, n_actions=self.n_actions, lr=lr, drop_rate = drop_rate)

        self.lr = lr
        self.baseline = 0.
        self.n_update = 0.
        self.training = True

    def lossfunc(self, rew, logprobs):

        loss = -( (rew.detach() - self.baseline) * logprobs).mean()

        if self.training:
           self.n_update += 1.
           self.baseline = 0. #+= (rew.detach().mean().item() - self.baseline) / (self.n_update)
        
        return loss