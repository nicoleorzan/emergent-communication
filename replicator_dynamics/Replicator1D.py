import numpy as np
from scipy.integrate import odeint
import copy

class Replicator1D():
    
    def __init__(self, A=[[1,2,0],[0,1,2],[2,0,1]]):
        self.A = A # by default RockPaperScissor
        
    def fit(self, x, i):
        return sum([x[j] * self.A[i][j] for j in range(len(x))])
            
    def avg_fit(self, x):
        return sum(x[i] * self.fit(x, i) for i in range(len(x)))
        
    def replicator(self, x, time=1000):
        return [x[i]*(np.matmul(self.A, x)[i] - np.matmul(x,np.matmul(self.A, x))) for i in range(len(x))]
        
    def loop(self, x0, time):
        x = x0
        assert(len(x0) == len(self.A[0]))

        self.x_time = np.zeros((time, len(x0)))
        for t in range(time):
            dx = self.replicator(x)
            for i in range(len(x)):
                x[i] += dx[i]
            self.x_time[t, :] = copy.deepcopy(x)

    def loop_ode(self, y0, time):
        y = odeint(self.replicator, y0, time)
        return y