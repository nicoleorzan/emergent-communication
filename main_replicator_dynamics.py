from replicator_dynamics.Replicator import Replicator1D, Replicator2D
from utils.Environments import Prisoner_Dilemma, CoordinationGame
import matplotlib.pyplot as plt
import numpy as np
import ternary

# Replicator Dynamics for one population

"""rep = Replicator1D()

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
"""

# Replicator Dynamics for two populations

""""
rep2 = Replicator2D()

time = 20000
y0 = [0.5, 0.01, 0.49]
x0 = [0.5, 0.25, 0.25]

rep2.loop_replicator(x0, y0, time)


plt.plot(rep2.x_time[:,0], rep2.x_time[:,1])
plt.xlabel("x0")
plt.ylabel("x1")
plt.savefig("rep_dyn_2d_x0_x1.png")
plt.show()

tup = [tuple(rep2.x_time[i,:]) for i in range(len(rep2.x_time[:,0]))]

figure, tax = ternary.figure(scale=1.0)
figure.set_size_inches(6, 5)
tax.boundary()
tax.gridlines(multiple=0.2, color="black")
tax.set_title("Plotting of sample trajectory data", fontsize=10)
points =  [tuple(rep2.x_time[i,:]) for i in range(len(rep2.x_time[:,0]))]

tax.plot(points, linewidth=2.0, label="Curve")
tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.legend()
tax.show()
tax.savefig("rep_dyn_circlating.png", bbox_inches='tight')



plt.plot(np.linspace(0, time, time), rep2.x_time[:,0], label="x0")
plt.xlabel("time")
plt.ylabel("x0")
plt.savefig("x0_time.png")
plt.show()

plt.plot(np.linspace(0, time, time), rep2.y_time[:,1], label="y1")
plt.xlabel("time")
plt.ylabel("y1")
plt.savefig("y1_time.png")
plt.show()"""


# Replicator Dynamics for two populations with MUTATION

#eta = -0.0
#A = np.array([[-1.,1.,eta],[eta,-1.,1.],[1.,eta,-1.]])
#B = np.array([[1.,-1.,-eta],[-eta,1.,-1.],[-1.,-eta,1.]])
#y0 = [0.5, 0.01, 0.49]
#x0 = [0.5, 0.25, 0.25]


game = Prisoner_Dilemma(T=0, R=-1, P=-2, S=-3)
game = CoordinationGame(2)
A = game.A
B = game.B

rep2 = Replicator2D(A, B)

time = 100000
y0 = [0.2, 0.8]
x0 = [0.9, 0.1]
tau = 1

rep2.loop_selection_mutation(x0, y0, tau, time)

print("final policies:")
print(rep2.x_time[time-1])
print(rep2.y_time[time-1])

print("game matrix:")
rep2.print_game_mmatrix()

fig, ax = plt.subplots(2, 1, figsize=(10,5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
fig.suptitle('x_t', y=0.95, fontsize=15)
ax[0].plot(np.linspace(0, time, time), rep2.ent_x)
ax[0].set_ylabel("Entropy x")
ax[0].grid()
ax[1].plot(np.linspace(0, time, time), rep2.ent_y)
ax[1].set_ylabel("Entropy y")
ax[1].grid()
plt.show()


fig, ax = plt.subplots(3,1, figsize=(10,5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
fig.suptitle('x_t', y=0.95, fontsize=15)
for i in range(len(x0)):
    ax[i].plot(np.linspace(0, time, time), rep2.x_time[:,i])
    ax[i].set_ylabel("x["+str(i)+"]")
    ax[i].grid()
#plt.savefig("x_time_mutation.png")
plt.show()


fig, ax = plt.subplots(3,1, figsize=(10,5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
fig.suptitle('x_t', y=0.95, fontsize=15)
for i in range(len(x0)):
    ax[i].plot(np.linspace(0, time, time), rep2.y_time[:,i])
    ax[i].set_ylabel("y["+str(i)+"]")
    ax[i].grid()
#plt.savefig("y_time_mutation.png")
plt.show()


plt.plot(rep2.x_time[:,0], rep2.x_time[:,1])
plt.xlabel("x0")
plt.ylabel("x1")
plt.savefig("rep_dyn_2d_x0_x1.png")
plt.show()


figure, tax = ternary.figure(scale=1.0)
figure.set_size_inches(6, 5)
tax.boundary()
tax.gridlines(multiple=0.2, color="black")
tax.set_title("Plotting of sample trajectory data", fontsize=10)
points =  [tuple(rep2.x_time[i,:]) for i in range(len(rep2.x_time[:,0]))]

tax.plot(points, linewidth=2.0, label="Curve")
tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()
tax.legend()
tax.show()
#tax.savefig("rep_dyn_mutation_converging.png", bbox_inches='tight')
