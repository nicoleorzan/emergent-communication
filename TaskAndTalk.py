import numpy as np
from agents.Agent import DiscAgent
from nltk.corpus import words
import random
from enum import Enum

"""class Features(Enum):
    #shape = _Shapes
    #color = _Colors
    #style = _Styles
    class _Shapes(Enum):
        triangle = 0
        square = 1
        circle = 2
        star = 3
    class _Colors(Enum):
        blue = 0
        green = 1
        red = 2
        purple = 3
    class _Styles(Enum):
        filled = 0
        dashed = 1
        dotted = 2
        solid = 3
"""

class TaskAndTalk:

    def __init__(self, config, Abot, Qbot):
        self.num_agents = 2
        self.config = config
        self.base_score = 0.05
        self.scores_step = 0.05
        self.returns = []
        self.R = 10 # base reward
        self.Abot = Abot
        self.Qbot = Qbot

        self.shapes = [0,1,2,3] #['triangle', 'square', 'circle', 'star']
        self.colors = [0,1,2,3] #['blue', 'green', 'red', 'purple']
        self.styles = [0,1,2,3] #['filled', 'dashed', 'dotted', 'solid']

        # self.features = Features

    def define_instance(self):
        return [random.choice(self.shapes), random.choice(self.colors), random.choice(self.styles)]

    def define_task(self):
        return sorted(random.sample([0, 1, 2], 2) )

    def unmap(self, T):
        if (T == 0):
            return [0,1]
        elif (T == 1):
            return [0,2]
        elif (T == 2):
            return [1,2]

    def define_ground_truth(self, I, T):
        #print("T=", T)
        #print("I=", I)
        gt = []
        for _, val in enumerate(T):
            #print("val=", val)
            gt.append(I[val])
            
        return gt

    def init_dialog(self):

        print("agents:", self.Abot.idx, self.Qbot.idx)

        self.T = self.define_task()
        self.I = self.define_instance()
        print("instance=", self.I, " task=", self.T)

        self.ground_truth = self.define_ground_truth(self.I, self.T)
        print("ground truth=", self.ground_truth)

        self.sQ = [self.T]
        self.sA = [self.I]

    def interaction_step(self):
        
        q_t = self.Qbot.ask(self.sQ)

        self.sQ.append(q_t)
        self.sA.append(q_t)

        a_t = self.Abot.answer(q_t)

        self.sQ.append(a_t)
        self.sA.append(a_t)


    def learning_loop(self, train_episodes):

        i = 0
        self.init_dialog()

        while(i < train_episodes):

            print("i=", i)
            self.interaction_step()

            i += 1
            print("\n")
        
        pred = self.Qbot.predict()
        _ = self.reward_function(self.ground_truth, pred)

        # qui update dei pesi


    def reward_function(self, ground_truth, pred):
        if (ground_truth == pred):
            return self.R
        return -10*self.R
        