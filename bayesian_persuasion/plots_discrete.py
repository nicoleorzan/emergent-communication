import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

fair = "fair"
#episodes = 15000
#df = pd.read_csv("dataset_"+str(episodes)+"_"+fair+"_discrete.csv")
df = pd.read_csv("tmp.csv")

#select a run
run = 0

#returns
df1 = df[df['run'] == run]
plt.plot(df1['i'], df1['rew'])
#plt.xticks([i for i in range(df.episode.max())])
plt.show()


#fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
#n_bins = 7
# We can set the number of bins with the `bins` kwarg
#axs.hist(df1['action'], bins=n_bins)
#plt.show()

# mean growth of returns
num_blocks = 100
data_per_block = int(len(df1['rew'])/num_blocks)
if (data_per_block < 1):
    print(" I cannot plot the smoothened reward for this run: not enough data")
else:
    cmean = []
    for i in range(0,num_blocks):
        cmean.append(np.mean(df1['rew'][i*data_per_block:i*data_per_block+data_per_block]))

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7, 5), gridspec_kw={'wspace': 0})

    fig.suptitle('Mean growth of Return over groups of ' + str(num_blocks) + ' episodes')

    axs.plot(np.linspace(0, num_blocks, num_blocks), cmean)
    axs.grid(b=True)
    plt.show()


# probabilities
fig, ax = plt.subplots(2, 2)
fig.suptitle("Stuff for run " + str(run))
ax[0,0].plot(df1['i'], df1["p(i|I)"], label='p(i|I)')
ax[0,1].plot(df1['i'], df1["p(c|I)"], label='p(c|I)')
ax[1,0].plot(df1['i'], df1["p(i|C)"], label='p(i|C)')
ax[1,1].plot(df1['i'], df1["p(c|C)"], label='p(c|C)')
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.legend()
plt.show()


# q values
"""num_probs=8
fig, ax = plt.subplots(num_probs, 2, figsize=(7,15))
fig.suptitle("Stuff for run " + str(run))
for i in range(num_probs):
    ax[i,0].plot(df1['i'], df1['q_0'+str(i)], label='q_0'+str(i))
    ax[i,1].plot(df1['i'], df1['q_1'+str(i)], label='q_1'+str(i))
    ax[i,0].grid()
    ax[i,1].grid()
    ax[i,0].set_ylim(0.5, 1.2)
    ax[i,1].set_ylim(-1, 0)
    ax[i,0].set_ylabel('q_0'+str(i))
    ax[i,1].set_ylabel('q_1'+str(i))
plt.show()
"""

if (len(df.run.unique())>1):

    # AVERAGES
    print("AVG REWARD PLOT")

    avg_rew = df.groupby('episode', as_index=False)['rew'].mean()
    avg_rew_std = df.groupby('episode', as_index=False)['rew'].std()
    markers, caps, bars = plt.errorbar(df.episode.unique(), avg_rew['rew'], avg_rew_std['rew'], label='rew')
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]
    plt.grid()
    plt.legend()
    plt.title("AVERAGE REWARD")
    plt.show()


    print("AVG PROBABILITIES PLOT")

    fig, ax = plt.subplots(2, 2)

    things = ['p(i|I)', 'p(c|I)', 'p(i|C)', 'p(c|C)']

    for i in range(2):
        for j in range(2):
            avg_rew = df.groupby('episode', as_index=False)[things[i*2+j]].mean()
            avg_rew_std = df.groupby('episode', as_index=False)[things[i*2+j]].std()
            markers, caps, bars = ax[i,j].errorbar(df.episode.unique(), avg_rew[things[i*2+j]], avg_rew_std[things[i*2+j]], label=things[i*2+j])
            [bar.set_alpha(0.1) for bar in bars]
            [cap.set_alpha(0.1) for cap in caps]
    plt.grid()
    plt.legend()
    plt.title("AVERAGE REWARD")
    plt.show()


