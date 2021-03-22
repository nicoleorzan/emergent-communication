import Agent
import random

class System:

    def __init__(self, num_agents, config, env):
        self.num_agents = num_agents
        self.config = config
        self.env = env
        self.adj_mat = None
        self.define_agents()

    def fill_adj_mat(self, adj_mat):
        self.adj_mat = adj_mat

    def define_agents(self):
        self.agents = {i:Agent.DiscAgent(self.config, self.env, i) for i in range(self.num_agents)}

    def get_reward(self, obs, act):
        if (obs == act):
            return 1
        return 0

    def choose(self, sender_idx):
        if (self.adj_mat):
            random.choice([i for i in range(self.num_agents) if self.adj_mat[sender_idx][i] == 1])
        else:
            return random.choice([i for i in range(self.num_agents) if i != sender_idx])

    def learning_step(self):
        sender_idx = random.choice(list(self.agents))
        receiver_idx = self.choose(sender_idx)
        print(sender_idx, receiver_idx)

        sender = self.agents[sender_idx]        
        receiver = self.agents[receiver_idx]

        obs = self.env.get_observation()
        
        mex = sender.generate_message(obs)
        action = receiver.perform_action(mex)
        
        rew = self.get_reward(obs, action)