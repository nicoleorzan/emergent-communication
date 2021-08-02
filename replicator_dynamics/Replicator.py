import numpy as np
from scipy.integrate import odeint
import copy    

class Replicator1D():
    
    def __init__(self, lr=0.1, A=[[1,2,0],[0,1,2],[2,0,1]]):
        self.A = A # by default RockPaperScissor
        self.lr = lr
        
    def replicator(self, x, time=1000):
        return [x[i]*(np.matmul(self.A, x)[i] - np.matmul(x,np.matmul(self.A, x))) for i in range(len(x))]

    def selection_mutation(self, x, tau, t=1000):
        _sum = np.sum([xi*np.log(xi) for xi in x])
        return [x[i]*self.lr/tau*(np.matmul(self.A, x)[i] - np.matmul(x,np.matmul(self.A, x))) - self.lr*x[i]*(np.log(x[i]) -_sum) for i in range(len(x))]
        
    def loop_replicator(self, x0, time):
        x = x0
        assert(len(x0) == len(self.A[0]))

        self.x_time = np.zeros((time, len(x0)))
        for t in range(time):
            dx = self.replicator(x)
            for i in range(len(x)):
                x[i] += dx[i]
            self.x_time[t, :] = copy.deepcopy(x)

    def loop_selection_mutation(self, x0, tau, time):
        x = x0
        assert(len(x0) == len(self.A[0]))

        self.x_time = np.zeros((time, len(x0)))
        for t in range(time):
            dx = self.selection_mutation(x, tau)
            for i in range(len(x)):
                x[i] += dx[i]
            self.x_time[t, :] = copy.deepcopy(x)

    def loop_ode(self, y0, time, mutation=False, tau=0.1):
        if (mutation == True):
            y = odeint(self.selection_mutation, y0, time, tau)
        else:
            y = odeint(self.replicator, y0, time)
        return y

class Replicator2D(Replicator1D):
    
    def __init__(self, A, B):
        self.A = A
        self.B = B
        #self.A = np.array([[1,0,0],[0,1,0],[0,0,1]])
        #self.B = np.array([[1,0,0],[0,1,0],[0,0,1]])

        self.lr = 0.01

    def print_game_mmatrix(self):
        for i in range(len(self.A[0])):
                print([np.array([self.A[i][j], self.B[i][j]]) for j in range(len(self.B[0]))])

    def replicatorx(self, x, y):
        dx = np.zeros(len(x))

        for i in range(len(x)):
            dx[i] = x[i]*(np.matmul(self.A, y)[i] - np.matmul(x,np.matmul(self.A, y)))
        return dx

    def replicatory(self, y, x):
        dy = np.zeros(len(y))

        for i in range(len(y)):
            dy[i] = y[i]*(np.matmul(x, self.B)[i] - np.matmul(x,np.matmul(self.B, y)))
        return dy

    def selection_mutationx(self, x, y, tau):
        dx = np.zeros(len(x))

        _sum = np.sum([xi*np.log(xi) for xi in x])
        for i in range(len(x)):
            dx[i] = self.lr/tau*x[i]*(np.matmul(self.A, y)[i] - np.matmul(x,np.matmul(self.A, y))) - self.lr*x[i]*(np.log(x[i]) -_sum) 
        return dx

    def selection_mutationy(self, y, x, tau):
        dy = np.zeros(len(y))

        _sum = np.sum([yi*np.log(yi) for yi in y])
        for i in range(len(y)):
            dy[i] = self.lr*tau*y[i]*(np.matmul(x, self.B)[i] - np.matmul(x,np.matmul(self.B, y))) - self.lr*y[i]*(np.log(y[i]) -_sum) 
        return dy
        
    def compute_entropy(self, x):
        H = 0
        for i in range(len(x)):
            H += (x[i]*np.log(x[i]))
        return H
            
    def loop_replicator(self, x0, y0, time):

        self.ent_x = []
        self.ent_y = []

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
                print("dx=",  self.replicatorx(x, y))
                print("dy=",  self.replicatory(y, x))
                print("\n")

            dx = self.replicatorx(x, y)
            for i in range(len(x)):
                x[i] += dx[i]*self.lr

            dy = self.replicatory(y, x)
            for i in range(len(y)):
                y[i] += dy[i]*self.lr

            self.x_time[t, :] = copy.deepcopy(x)
            self.y_time[t, :] = copy.deepcopy(y)

            self.ent_x.append(self.compute_entropy(x))
            self.ent_y.append(self.compute_entropy(y))


    def loop_selection_mutation(self, x0, y0, tau, time):

        self.ent_x = []
        self.ent_y = []

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
                print("dx=",  self.selection_mutationx(x, y, tau))
                print("dy=",  self.selection_mutationy(y, x, tau))
                print("\n")

            dx = self.selection_mutationx(x, y, tau)
            for i in range(len(x)):
                x[i] += dx[i]*self.lr

            dy = self.selection_mutationy(y, x, tau)
            for i in range(len(y)):
                y[i] += dy[i]*self.lr

            self.x_time[t, :] = copy.deepcopy(x)
            self.y_time[t, :] = copy.deepcopy(y)

            self.ent_x.append(self.compute_entropy(x))
            self.ent_y.append(self.compute_entropy(y))