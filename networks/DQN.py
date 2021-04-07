import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    
    def __init__(self, lr, input_dim, num_actions, n_intermed_nodes):

        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.intermed_nodes = n_intermed_nodes
        self.fc1 = nn.Linear(self.input_dim, self.intermed_nodes)
        self.fc2 = nn.Linear(self.intermed_nodes, self.num_actions)
        self.device = t.device("cpu")
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
    def forward(self, s):
        x = F.relu(self.fc1(s))
        return self.fc2(x) # not activated