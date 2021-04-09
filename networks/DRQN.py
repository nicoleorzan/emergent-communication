import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DRQN(nn.Module):
    
    def __init__(self, lr, input_dim, num_actions, n_layers, hidden_layer=60):
        
        super(DRQN, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_layer = hidden_layer
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(self.input_dim, self.hidden_layer, self.n_layers, batch_first=True)  
        self.fc = nn.Linear(self.hidden_layer, self.num_actions)

        self.device = t.device("cpu")
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
    def forward(self, batch):

        batch_size = batch.size(0)
        print("batch size=", batch_size)
        print("batch=", batch)
        #sequence_length = batch.size(1)
        #print("seq len=", sequence_length)

        hidden = t.zeros(1*self.n_layers, batch_size, self.hidden_layer, device=self.device, dtype=t.float)

        #print("hidden=", hidden)
        out, _ = self.rnn(batch, hidden)
        #print("out=", out)
        #out = out.reshape(out.shape[0], -1)
        #print("out reshape=", out)
        out = self.fc(out)
        #print("final=", out)
        return out