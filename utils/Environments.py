import random

class BaseEnv:

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_observation(self):
        pass

    def get_reward(self):
        pass

class DiscEnv(BaseEnv):

    def __init__(self, num_obs, num_actions):
        super().__init__(num_actions)
        self.num_obs = num_obs

    def get_observation(self):
        return random.randint(0, self.num_obs-1) #random.choice([i for i in range(self.num_obs)])

    def get_reward(self, obs, act):
        if (obs == act):
            return 1
        return -10
"""
class ContEnv(BaseEnv):

    def __init__(self, range, num_actions):
        super().__init__(num_actions)
        self.min, self.max = range

    def get_observation(self):
        return random.uniform(self.min, self.max) 

    def get_reward(self, obs, act):
        if (obs < act):
            if (act-obs < 1):
                return 1
            else:
                return 0
        return 0
"""