class Config:

    def __init__(self, _dict):
        self.epsilon = _dict["epsilon"]
        self.lr = _dict["lr"]
        
        self.batch_size = 50
        self.learn_start = 100
        self.update_freq = 1
        self.experience_replay_size = 1000
        
        self.target_net_update_freq = 100
        
        self.nsteps = 1