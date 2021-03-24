import numpy as np

class BaseAgent():

    def __init__(self, config, env):

        self.lr = config.lr
        self.epsilon = config.epsilon

        self.num_actions = env.num_actions

        self.rewards = []

    def take_action(self):
        pass

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
