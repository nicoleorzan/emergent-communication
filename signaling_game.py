from games.SignalingGame import SignalingGame
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

#num_agents = 2
num_obs = 3
words_length = num_obs # can be >= obs_length
dynamics=True

s = SignalingGame(words_length=words_length, learning_rate=0.1, epsilon=0.05,
 epsilon_step=0.1, dynamics=dynamics, agents_fixed=True, game_cooperative=False)
num_agents = s.num_agents

time = 10000
s.train(time)

agents = [s.sender, s.receiver]
if (dynamics):

    s_dynamics = s.sender_states_t

    fig, ax = plt.subplots(num_obs, num_obs, figsize=(4*num_obs,2*num_obs), sharex=True, sharey=True, gridspec_kw={'wspace':0.1, 'hspace': 0.2})
    fig.suptitle('Sender Dynamics', y=0.9, fontsize=15)
        
    for i in range(num_obs):
        for j in range(num_obs):
            ax[i,j].plot(s_dynamics[i,j,:])

            ax[i,j].set_xlabel("Obs="+str(i), fontsize=13)
            ax[i,j].set_ylabel("Mex="+str(j), fontsize=13)
            ax[i,j].grid()


    s_dynamics = s.receiver_states_t

    fig, ax = plt.subplots(num_obs, num_obs, figsize=(4*num_obs,2*num_obs), sharex=True, sharey=True, gridspec_kw={'wspace':0.1, 'hspace': 0.2})
    fig.suptitle('Receiver Dynamics', y=0.9, fontsize=15)
        
    for i in range(num_obs):
        for j in range(num_obs):
            ax[i,j].plot(s_dynamics[i,j,:])

            ax[i,j].set_xlabel("Mex="+str(i), fontsize=13)
            ax[i,j].set_ylabel("Act="+str(j), fontsize=13)
            ax[i,j].grid()
        
    plt.show()




fig, ax = plt.subplots(num_agents, figsize=(2*num_obs,3.5*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
fig.suptitle('Sender Matrices', y=0.9, fontsize=15)
#ax[0].set_ylim(-0.01,1.01)

for j in range(num_agents):
    
    for i in range(num_obs):
        ax[j].plot(agents[j].get_q_sender()[i], label="obs idx="+str(i))

    ax[j].set_xlabel("Word", fontsize=15)
    ax[j].set_ylabel("Agent "+str(j)+"\nWord Usage Probability", fontsize=13)
    ax[j].grid()
    ax[j].legend(title="Idx observation")
    
ax[num_agents-1].set_xticks([i for i in range(words_length)])
plt.show()


fig, ax = plt.subplots(num_agents, figsize=(2*num_obs,3.5*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
fig.suptitle('Receiver Matrices', y=0.9, fontsize=15)
#ax[0].set_ylim(-0.01,1.01)

for j in range(num_agents):
    
    for i in range(words_length):
        ax[j].plot(agents[j].get_q_receiver()[i], label=str(i))

    ax[1].set_xlabel("Possible Actions", fontsize=15)
    ax[j].set_ylabel("Agent "+str(j)+"\nAction Probability", fontsize=15)
    ax[j].set_xticks([i for i in range(num_obs)])
    ax[j].grid()
ax[num_agents-1].legend(title="Idx received message")
plt.show()
