from replicator_dynamics.Replicator1D import Replicator1D
import matplotlib.pyplot as plt
import numpy as np

rep = Replicator1D()

max_time = 100
time = np.arange(0, max_time, .1)

space = np.linspace(0.05, 0.95, 10)
for y_0 in space:
    for y_1 in np.linspace(0.05, 0.95-y_0, 10):
        y0 = [y_0, y_1, 1.-(y_0 + y_1)]
        print("y0=", y0)
        y = rep.loop_ode(y0, time)
        plt.plot(y[:,0], y[:,1])
plt.show()