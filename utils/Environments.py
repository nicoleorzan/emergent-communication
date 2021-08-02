import random
import numpy as np

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

class MONFG():

    def __init__(self):
        self.num_obs = 3
        self.num_actions = 3
        self.A = np.array([[4,1,2],[3,3,1],[1,2,1]])
        self.B = np.array([[1,2,1],[1,2,2],[2,1,3]])

    def get_observation(self):
        return random.randint(0, self.num_obs-1)

    def get_reward(self, obs, act): #0=rock, 1=paper, 2=scissot
        sender_rew = 0
        receiver_rew = 0

        sender_rew = self.A[obs, act]
        receiver_rew = self.B[obs, act]
        
        return sender_rew, receiver_rew

class RockPaperScissor():

    def __init__(self):
        self.num_obs = 3
        self.num_actions = 3

    def get_observation(self):
        return random.randint(0, self.num_obs-1)

    def get_reward(self, obs, act): #0=rock, 1=paper, 2=scissot
        sender_rew = 0
        receiver_rew = 0

        sott = act - obs
        if (sott == 1 or sott == -2):
            sender_rew = -1
            receiver_rew = 1
        elif (sott == -1 or sott == 2): # rock, paper, scissor
            sender_rew = 1
            receiver_rew = -1

        return sender_rew, receiver_rew

class Prisoner_Dilemma():

    def __init__(self, T=2, R=1, P=0, S=-1): # (T > R > P > S). 
        self.num_obs = 2
        self.num_actions = 2
        self.T = T
        self.R = R
        self.P = P
        self.S = S
        assert(self.T > self.R) # defection is the dominant strategy for both agents. 
        assert(self.R > self.P) # mutual cooperation is superior to mutual defection,
        assert(self.P > self.S) # defection is the dominant strategy for both agents. 
        self.A = np.array([[self.R, self.S],[self.T, self.P]])
        self.B = np.array([[self.R, self.T],[self.S, self.P]])

    def print_matrices(self):
        for i in range(len(self.A)):
            print([self.A[i,0],self.B[i,0]], [self.A[i,1],self.B[i,1]])

    def get_observation(self):
        return random.randint(0, self.num_obs-1)

    def get_reward(self, obs, act):

        sender_rew = self.A[obs, act]
        receiver_rew = self.B[obs, act]
        
        return sender_rew, receiver_rew

class CoordinationGame():

    def __init__(self, num_obs):
        self.num_obs = num_obs
        self.A = np.zeros((num_obs, num_obs))
        self.B = np.zeros((num_obs, num_obs))
        for i in range(self.num_obs):
            self.A[i][i] = 1
            self.B[i][i] = 1

    def print_matrices(self):
        for i in range(self.num_obs):
            print([ [self.A[i,j],self.B[i,j]] for j in range(self.num_obs)])
        print("\n")

    def get_observation(self):
        return random.randint(0, self.num_obs-1)

    def get_reward(self, obs, act):

        sender_rew = self.A[obs, act]
        receiver_rew = self.B[obs, act]
    
        return sender_rew, receiver_rew

class ContEnv(BaseEnv):

    def __init__(self, range, max_obs_num=5):
        np.random.seed(123)
        #super().__init__(num_actions)
        self.min, self.max = range
        self.max_obs_num = max_obs_num
        self.dmin = 0.2
        self.limit = 1000

    def get_observation(self):
        #print("get observation")
        num = random.choice(range(2, self.max_obs_num))
        obs = [random.uniform(self.min, self.max)]
        #print("obs=", obs)
        #print("num=", num)
        i = 0
        while (len(obs) < num):
            if (i > self.limit):
                break

            o_i = random.uniform(self.min, self.max)
            #print("o_i=", o_i)

            while (any( abs(o_i - o_j)<self.dmin for o_j in obs)):
                o_i = random.uniform(self.min, self.max)
            
            obs.append(o_i)
            i += 1
            print("done here")

        #print("final object=", obs)
        return sorted(obs)

    """def get_reward(self, obs, act):
        if (obs < act):
            if (act-obs < 1):
                return 1
            else:
                return 0
        return 0
    """