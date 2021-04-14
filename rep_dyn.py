from replicator_dynamics.Replicator import Replicator1D, Replicator2D
import matplotlib.pyplot as plt
import numpy as np

# Replicator Dynamics for one population

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



# Replicator Dynamics for two populations

rep2 = Replicator2D()

time = 50000
y0 = [0.5, 0.01, 0.49]
x0 = [0.5, 0.25, 0.25]

rep2.loop(x0, y0, time)

plt.plot(rep2.x_time[:,0], rep2.x_time[:,1])
plt.show()

plt.plot(np.linspace(0, time, time), rep2.x_time[:,0], label="x0")
plt.show()

plt.plot(np.linspace(0, time, time), rep2.y_time[:,1], label="y1")
plt.show()