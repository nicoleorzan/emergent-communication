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

class Replicator2D():
    
    def __init__(self):
        self.eta = -0.0
        self.A = np.array([[-1.,1.,self.eta],[self.eta,-1.,1.],[1.,self.eta,-1.]])
        self.B = np.array([[1.,-1.,-self.eta],[-self.eta,1.,-1.],[-1.,-self.eta,1.]])
        self.lr = 0.01
        assert( ((self.A + self.B) == np.zeros((3,3))).all())

    def stepx(self, x, y):
        dx = np.zeros(len(x))

        for i in range(len(x)):
            #dx[i] = x[i]*(self.fit1(y, i) - self.avg_fit1(x,y)) 
            dx[i] = x[i]*(np.matmul(self.A, y)[i] - np.matmul(x,np.matmul(self.A, y)))
        return dx

    def stepy(self, y, x):
        dy = np.zeros(len(y))

        for i in range(len(y)):
            #dy[i] = y[i]*(self.fit2(x, i) - self.avg_fit1(y,x)) 
            dy[i] = y[i]*(np.matmul(x, self.B)[i] - np.matmul(x,np.matmul(self.B, y)))
        return dy
        
    def loop(self, x0, y0, time):

        assert(len(x0) == len(self.A[0]))
        assert(len(y0) == len(self.B[0]))
        x = x0
        y = y0

        self.x_time = np.zeros((time, len(x)))
        self.y_time = np.zeros((time, len(y)))

        for t in range(time):
            if (t%1000 == 0):
                print("t=", t)
                print("x=",x)
                print("y=",y)
                print("tmp equilibrium: x=",np.argmax(x), "y=", np.argmax(y))
                print("fitnss for x=", self.A[np.argmax(x)][np.argmax(y)])
                print("fitnss for y=", self.B[np.argmax(x)][np.argmax(y)])
                print("dx=",  self.stepx(x, y))
                print("dy=",  self.stepy(y, x))
                print("\n")

            dx = self.stepx(x, y)
            for i in range(len(x)):
                x[i] += dx[i]*self.lr

            dy = self.stepy(y, x)
            for i in range(len(y)):
                y[i] += dy[i]*self.lr

            self.x_time[t, :] = copy.deepcopy(x)
            self.y_time[t, :] = copy.deepcopy(y)