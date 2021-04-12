from games.SignalingGame import SignalingGame
import matplotlib.pyplot as plt

num_agents = 2
num_obs = 5
words_length = num_obs # can be >= obs_length

s = SignalingGame(num_agents=num_agents, num_obs=num_obs, epsilon=0.1, learning_rate=0.9, words_length=words_length)

s.train(1000)
s.test_loop(100)
s.print_matrices(0)

agents = s.dict_agents


fig, ax = plt.subplots(num_agents, figsize=(2*num_obs,3.5*num_agents), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
fig.suptitle('Sender Matrices', y=0.9, fontsize=15)
ax[0].set_ylim(-0.01,1.01)

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
ax[0].set_ylim(-0.01,1.01)

for j in range(num_agents):
    
    for i in range(words_length):
        ax[j].plot(agents[j].get_q_receiver()[i], label=str(i))

    ax[1].set_xlabel("Possible Actions", fontsize=15)
    ax[j].set_ylabel("Agent "+str(j)+"\nAction Probability", fontsize=15)
    ax[j].set_xticks([i for i in range(num_obs)])
    ax[j].grid()
ax[num_agents-1].legend(title="Idx received message")
plt.show()
