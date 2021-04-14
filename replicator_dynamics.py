from replicator_dynamics.Replicator1D import Replicator1D
import matplotlib.pyplot as plt
import numpy as np

rep = Replicator1D()

max_time = 100
time = np.arange(0, max_time, .1)
y0 = [0.1, 0.8, 0.1]

y = rep.loop_ode(y0, time)
plt.plot(y[:,0], y[:,1])
plt.show()