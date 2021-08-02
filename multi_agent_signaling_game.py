from games.PopSignalingGame import PopSignalingGame
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils.Environments import Prisoner_Dilemma, CoordinationGame, RockPaperScissor

num_agents = 5
G = nx.erdos_renyi_graph(n = num_agents, p = 0.999)
#G.remove_nodes_from(list(nx.isolates(G)))

#nx.draw(G,with_labels = True)
#plt.savefig("system_communication_network.png")
#plt.show()

A = nx.adjacency_matrix(G)

adjmat = np.asarray(A.todense())
type(adjmat)


"""num_obs = 4
num_messages = 3

bottleneck = ''
if (num_messages < num_obs):
    bottleneck = '_bottleneck'

num_actions = num_obs
game = CoordinationGame(num_obs = num_actions)
"""

game = Prisoner_Dilemma()
game_name = 'prisoner'

num_obs = 2
num_messages = num_obs

bottleneck = ''
if (num_messages < num_obs):
    bottleneck = '_bottleneck'

num_actions = num_obs
game.print_matrices()

T = 0.1

sy = PopSignalingGame(game = game, num_agents=num_agents, num_obs = num_obs, num_messages = num_messages, num_actions = num_actions, 
        epsilon=0.05, epsilon_step = 0.1, learning_rate = 0.1, T=T)
sy.give_adj_matrix(adjmat)

#x0_sender = [0.333, 0.333, (1-0.333*2)]
#x0_sender = [0.25, 0.25, 0.25, 0.25]
#x0_receiver = [0.25, 0.25, 0.25, 0.25]

x0_sender = [0.5, 0.5]
x0_receiver = [0.5, 0.5]

assert(len(x0_sender) == num_messages)
assert(len(x0_receiver) == num_obs)

sy.init_policy(x0_sender, x0_receiver)

steps = 10000
sy.perform_loop(steps)

agents = sy.return_agents_dict()

print("entropy senders")
for k,v in agents.items():
    print(v.compute_entropy_comm())

print("entropy receivers")
for k,v in agents.items():
    print(v.compute_entropy_act())


print("action probabilitis: senders")
for k,v in agents.items():
    pi_send = v.get_pi_sender()
    for o in range(num_obs):
        print("obs=", o, "probs:", pi_send[o,:])
    print("\n")

print("action probabilitis: receivers")
for k,v in agents.items():
    pi_rec = v.get_pi_receiver()
    for m in range(num_messages):
        print("mex=", m, "probs:", pi_rec[m,:])
    print("\n")

# POLICIES
#=========================================
fig, ax = plt.subplots(num_agents, figsize=(6,3.3*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})

fig.suptitle("Sender Matrices", y=0.9, fontsize=13)

ax[0].set_ylim(-0.01,1.01)

for j in range(num_agents):
    
    for i in range(num_obs):
        ax[j].plot(agents[j].get_pi_sender()[i], label="obs idx="+str(i))

    ax[j].set_xlabel("Word", fontsize=15)
    ax[j].set_ylabel("Agent "+str(j)+"\nWord Prob", fontsize=13)
    ax[j].set_ylim(0,1)
    ax[j].grid()
    
    ax[j].legend(title="Idx observation")
    
ax[num_agents-1].set_xticks([i for i in range(num_messages)])
#plt.show()
plt.savefig("sender_policy"+game_name+bottleneck+".png")


#=========================================
fig, ax = plt.subplots(num_agents, figsize=(6,3.3*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})

fig.suptitle("Receiver Matrices", y=0.9, fontsize=13)

ax[0].set_ylim(-0.01,1.01)

for j in range(num_agents):
    
    for i in range(num_messages):
        ax[j].plot(agents[j].get_pi_receiver()[i], label=str(i))

    ax[j].set_ylabel("Agent "+str(j)+"\nAction Prob", fontsize=15)
    ax[j].set_xticks([i for i in range(num_obs)])
    ax[j].grid()
    ax[j].set_ylim(0,1)
    ax[j].legend(title="Idx received message")

ax[num_agents-1].set_xlabel("Possible Actions", fontsize=15)
#plt.show()
plt.savefig("receiver_policy"+game_name+bottleneck+".png")



# ENTROPIES
#=========================================

fig, ax = plt.subplots(num_agents, figsize=(6,3.5*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})

fig.suptitle("Sender Entropy", y=0.9, fontsize=13)

print("sender entropy")
for j in range(num_agents):
    ent = agents[j].compute_entropy_comm()
    print(ent)
    ax[j].plot(np.arange(0, len(ent), 1), ent, 'o')
    ax[j].set_xticks([i for i in range(len(ent))])

    ax[j].set_xlabel("Word", fontsize=15)
    ax[j].set_ylabel("Agent "+str(j)+"\nWord Entropy", fontsize=13)
    ax[j].set_ylim(0,2)
    
    #ax[j].legend(title="Idx observation")
    
    #ax[j].set_xticks([i for i in range(num_messages)])
    ax[j].grid()

#plt.show()
plt.savefig("sender_entropy"+game_name+bottleneck+".png")

#=========================================

fig, ax = plt.subplots(num_agents, figsize=(7,3.5*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})

fig.suptitle("Receiver Entropy", y=0.9, fontsize=13)

print("receiver entropy")

for j in range(num_agents):
    ent = agents[j].compute_entropy_act()
    print(ent)
    ax[j].plot(np.arange(0, len(ent), 1), ent, 'o')

    ax[j].set_xlabel("Word", fontsize=15)
    ax[j].set_ylabel("Agent "+str(j)+"\nAction Entropy", fontsize=13)
    ax[j].set_ylim(0,2)
    ax[j].grid()
    
    #ax[j].legend(title="Idx messaage")
    
    #ax[j].set_xticks([i for i in range(num_actions)])
#plt.show()
plt.savefig("receiver_entropy"+game_name+bottleneck+".png")



#====== PI COMM time
fig, ax = plt.subplots(num_agents, num_obs, figsize=(18,3.2*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})

fig.suptitle("Sender Policy in Time", y=0.9, fontsize=13)

for a in range(num_agents):
    
    for i in range(num_obs):
        for j in range(num_messages):
            ax[a,i].plot(np.linspace(0, steps, steps), agents[a].pi_comm_t[:,i,j], label="obs = "+str(i)+", mex="+str(j))

        ax[a,i].set_ylabel("Agent "+str(a)+"\nProb mex " + str(i), fontsize=13)
        ax[a,i].grid()
        ax[a,i].legend(title="Idx received obs")

        ax[a,i].set_xlabel("Time", fontsize=15)
#plt.show()
plt.savefig("sender_policy_time"+game_name+bottleneck+".png")

#====== PI ACT
fig, ax = plt.subplots(num_agents, num_messages, figsize=(18,3.2*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})

fig.suptitle("Receiver Policy in Time", y=0.9, fontsize=13)

for a in range(num_agents):
    
    for i in range(num_messages):
        for j in range(num_actions):
            ax[a,i].plot(np.linspace(0, steps, steps), agents[a].pi_act_t[:,i,j], label="mex = "+str(i)+", act="+str(j))

        ax[a,i].set_ylabel("Agent "+str(a)+"\nProb action " + str(i), fontsize=13)
        ax[a,i].grid()
        ax[a,i].legend(title="Idx received message")

        ax[a,i].set_xlabel("Time", fontsize=15)
#plt.show()
plt.savefig("receiver_policy_time"+game_name+bottleneck+".png")