import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

fair = "fair"
episodes = 5000
df = pd.read_csv("dataset2_"+str(episodes)+"_"+fair+"_locale.csv")
#df = pd.read_csv("dataset2_"+str(episodes)+"_"+fair+"_lr03.csv")
runs = df.run.max() + 1

print("PROBABILITIES PLOT RUN 0")

fig, ax = plt.subplots(2, 2)
fig.suptitle("PROBABILITIES")

for run in range(1):
    df1 = df[df['run'] == run]
    ax[0,0].set_title("p(i|I)")
    ax[0,0].plot(df1['episode'],df1['p(i|I)'], label='p(i|I)', color='blue')
    ax[0,1].set_title("p(i|C)")
    ax[0,1].plot(df1['episode'],df1['p(i|C)'], label='p(i|C)', color='red')
    ax[1,0].set_title("p(c|I)")
    ax[1,0].plot(df1['episode'],df1['p(c|I)'], label='p(c|I)', color='orange')
    ax[1,1].set_title("p(c|C)")
    ax[1,1].plot(df1['episode'],df1['p(c|C)'], label='p(c|C)', color='green')
ax[0,0].set_ylim(-0.05,1.05)
ax[0,1].set_ylim(-0.05,1.05)
ax[1,0].set_ylim(-0.05,1.05)
ax[1,1].set_ylim(-0.05,1.05)
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()




# conto num di volte che effettivamente converge al valore giusto
converged_list = []
precision = 0.05
print("precision=", precision)

for ep in df.episode.unique():
    df1 = df[df['episode'] == ep]
    conv_df = df1[abs(df1["avg_rew"] - 1.0) < precision]
    converged_list.append(len(conv_df))

    if (ep == df.episode.max()):
        non_conv_runs = df1[abs(df1["avg_rew"] - 1.0) >= precision].run.unique()
        conv_runs = df1[abs(df1["avg_rew"] - 1.0) < precision].run.unique()

print("Runs that did not converge:", non_conv_runs)
print("Number of those:", len(non_conv_runs))
print("Number of converged runs:", converged_list[-1], "over", (df.run.max()+1))
print("Final proportion of converged runs: ", converged_list[-1]/(df.run.max()+1))
plt.plot(np.linspace(0, len(converged_list), len(converged_list)), converged_list/(df.run.max()+1))
plt.xlabel("Episodes")
plt.ylabel("# Converged Runs")
plt.grid()
plt.title("Percentage of converged runs for the number of episodes")
plt.show()


fig, ax = plt.subplots(2, 2)

non_conv_df = df[df["run"].isin(non_conv_runs)]
conv_df = df[df["run"].isin(conv_runs)]

for r in non_conv_df.run.unique():
    df_tmp = non_conv_df[non_conv_df['run'] == r]
    ax[0,0].plot(df_tmp['episode'], df_tmp['avg_rew'])
    ax[0,1].plot(df_tmp['episode'], df_tmp['loss'])

for r in conv_df.run.unique():
    df_tmp = conv_df[conv_df['run'] == r]
    ax[1,0].plot(df_tmp['episode'], df_tmp['avg_rew'])
    ax[1,1].plot(df_tmp['episode'], df_tmp['loss'])

ax[0,0].set_ylabel("Non Converged\nReward")
ax[1,0].set_ylabel("Converged\nReward")
ax[0,1].set_ylabel("Loss")
ax[1,1].set_ylabel("Loss")
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()




fig, ax = plt.subplots(2, 2)
num_runs = 1
print(non_conv_df.run.unique())
fig.suptitle("Non Converged Runs" + str(non_conv_df.run.unique()[0:num_runs]))

colors = ["blue", "red", "orange", "green", "black"]
for idx, r in enumerate(non_conv_df.run.unique()[0:num_runs]):
    df_tmp = non_conv_df[non_conv_df['run'] == r]
    ax[0,0].plot(df_tmp['episode'], df_tmp['p(i|I)']+idx/100, label='p(i|I)', color=colors[idx])
    ax[0,1].plot(df_tmp['episode'], df_tmp['p(c|I)']+idx/100, label='p(c|I)', color=colors[idx])
    ax[1,0].plot(df_tmp['episode'], df_tmp['p(i|C)']+idx/100, label='p(i|C)', color=colors[idx])
    ax[1,1].plot(df_tmp['episode'], df_tmp['p(c|C)']+idx/100, label='p(c|C)', color=colors[idx])

ax[0,0].set_ylabel("p(i|I)")
ax[0,1].set_ylabel("p(c|I)")
ax[1,0].set_ylabel("p(i|C)")
ax[1,1].set_ylabel("p(c|C)")
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()

fig1, ax1 = plt.subplots(2, 2)
fig1.suptitle("Converged Runs" + str(conv_df.run.unique()[0:num_runs]))

for idx, r in enumerate(conv_df.run.unique()[0:num_runs]):
    df_tmp = conv_df[conv_df['run'] == r]
    ax1[0,0].plot(df_tmp['episode'], df_tmp['p(i|I)']+idx/100, label='p(i|I)', color=colors[idx])
    ax1[0,1].plot(df_tmp['episode'], df_tmp['p(c|I)']+idx/100, label='p(c|I)', color=colors[idx])
    ax1[1,0].plot(df_tmp['episode'], df_tmp['p(i|C)']+idx/100, label='p(i|C)', color=colors[idx])
    ax1[1,1].plot(df_tmp['episode'], df_tmp['p(c|C)']+idx/100, label='p(c|C)', color=colors[idx])

ax1[0,0].set_ylabel("p(i|I)")
ax1[0,1].set_ylabel("p(c|I)")
ax1[1,0].set_ylabel("p(i|C)")
ax1[1,1].set_ylabel("p(c|C)")
ax1[0,0].grid()
ax1[0,1].grid()
ax1[1,0].grid()
ax1[1,1].grid()
plt.show()




print("LOSS PLOT")
plt.grid()
plt.title("LOSS PLOT")
for run in range(runs):
    df1 = df[df['run'] == run]
    plt.plot(df1["episode"], df1["loss"])
plt.yscale('log')
plt.show()


print("REWARDS PLOT")
plt.title("REWARDS PLOT")
for r in range(runs):
    df1 = df[df['run'] == r]
    plt.plot(df1['episode'], df1['avg_rew'])
plt.grid()
plt.show()


print("AVG REWARD PLOT")

avg_rew = df.groupby('episode', as_index=False)['avg_rew'].mean()
avg_rew_std = df.groupby('episode', as_index=False)['avg_rew'].std()
markers, caps, bars = plt.errorbar(df.episode.unique(), avg_rew['avg_rew'], avg_rew_std['avg_rew'], label='avg_rew')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]
plt.grid()
plt.legend()
plt.title("AVERAGE REWARD")
plt.show()







print("PROBABILITIES PLOT")

fig, ax = plt.subplots(2, 2)
fig.suptitle("PROBABILITIES")

for run in range(runs):
    df1 = df[df['run'] == run]
    ax[0,0].set_title("p(i|I)")
    ax[0,0].plot(df1['episode'],df1['p(i|I)'], label='p(i|I)', color='blue')
    ax[0,1].set_title("p(i|C)")
    ax[0,1].plot(df1['episode'],df1['p(i|C)'], label='p(i|C)', color='red')
    ax[1,0].set_title("p(c|I)")
    ax[1,0].plot(df1['episode'],df1['p(c|I)'], label='p(c|I)', color='orange')
    ax[1,1].set_title("p(c|C)")
    ax[1,1].plot(df1['episode'],df1['p(c|C)'], label='p(c|C)', color='green')
ax[0,0].set_ylim(-0.05,1.05)
ax[0,1].set_ylim(-0.05,1.05)
ax[1,0].set_ylim(-0.05,1.05)
ax[1,1].set_ylim(-0.05,1.05)
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()


print("POSTERIORS PLOT")

fig, ax = plt.subplots(2, 2)
fig.suptitle("POSTERIORS")

for run in range(runs):
    df1 = df[df['run'] == run]
    ax[0,0].set_title("p_I_i")
    ax[0,0].plot(df1['episode'],df1['p_I_i'], label='p_I_i', color='blue')
    ax[0,1].set_title("p_I_c")
    ax[0,1].plot(df1['episode'],df1['p_I_c'], label='p_I_c', color='red')
    ax[1,0].set_title("p_C_i")
    ax[1,0].plot(df1['episode'],df1['p_C_i'], label='p_C_i', color='orange')
    ax[1,1].set_title("p_C_c")
    ax[1,1].plot(df1['episode'],df1['p_C_c'], label='p_C_c', color='green')
ax[0,0].set_ylim(-0.05,1.05)
ax[0,1].set_ylim(-0.05,1.05)
ax[1,0].set_ylim(-0.05,1.05)
ax[1,1].set_ylim(-0.05,1.05)
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()
plt.show()

