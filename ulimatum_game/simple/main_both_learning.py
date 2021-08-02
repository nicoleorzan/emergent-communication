import matplotlib.pyplot as plt
from agents import Sender, Receiver
import pandas as pd
import seaborn as sns

n_actions = 10
epsilon = 0.2
lr = 0.1
n_thresholds = 1
thresholds = [0.3] #[(i/n_thresholds) for i in range(1,n_thresholds)]

num_episodes = 10000
ep_step = 10


def get_reward(resp, p):
    if (resp == 1):
        return 1.-p, p
    else: 
        return 0., 0.


def train(proposer, receiver, run_n, n_episodes, df):
    
    for ep in range(1,n_episodes+1):
        
        prop_idx = proposer.epsilon_greedy_proposal()
        prop = proposer.act(prop_idx)
        resp = receiver.epsilon_greedy_recv(prop_idx)

        rew_proposer, rew_receiver = get_reward(resp, prop)
        proposer.TD_prop(prop_idx, rew_proposer)
        receiver.TD_recv(prop_idx, resp, rew_receiver)
        proposer.softmax()
        #print("prop=", prop, "resp=", resp, "rew_prop=", rew_proposer, "rew_recv=", rew_receiver)

        if (ep%ep_step == 0):
            df.loc[int(run_n*num_episodes/ep_step+ep/ep_step)] = [run_n, ep, receiver.threshold, epsilon, rew_proposer, rew_receiver, "train"] + list(proposer.q_prop) + list(receiver.q_recv[:,1])


def eval(proposer, receiver, run_n, n_episodes_eval, df):

    proposer.train = False
    receiver.train = False

    for ep in range(1,n_episodes_eval+1):
        
        prop_idx = proposer.epsilon_greedy_proposal()
        prop = proposer.act(prop_idx)
        resp = receiver.epsilon_greedy_recv(prop_idx)

        rew_proposer, rew_receiver = get_reward(resp, prop)

        if (ep%ep_step == 0):
            print(num_episodes + run_n*num_episodes/ep_step+ep/ep_step)
            df.loc[int(num_episodes + run_n*num_episodes/ep_step+ep/ep_step)] = [run_n, num_episodes + ep, receiver.threshold, epsilon, rew_proposer, rew_receiver, "eval"] + list(proposer.q_prop) + list(receiver.q_recv[:,1])



def main(n_episodes):

    df = pd.DataFrame(columns = ['run', 'episode', 'threshold', 'epsilon', 'rew_proposer', 'rew_receiver', 'train?'] \
        + ["prop_" + str(i/n_actions) for i in range(n_actions)] +
          ["recv_" + str(i/n_actions) for i in range(n_actions)] )

    for run, threshold in enumerate(thresholds):

        proposer = Sender(idx = 0, lr = lr, n_actions=n_actions, epsilon=epsilon)
        receiver = Receiver(idx = 1, lr = lr, threshold=threshold, n_states=n_actions, epsilon=epsilon)

        print("Run number=", run, " with threshold=", threshold)
        print("Train")
        train(proposer, receiver, run, n_episodes, df)
        print("Eval")
        eval(proposer, receiver, run_n = 1, n_episodes_eval=0, df=df)

    return df



if __name__ == "__main__":

    df = main(n_episodes = num_episodes)

    print(df.tail(10))
    df1 = df[df["episode"] == df.episode.max()]
    df1 = df1[df1["train?"] == "train"]

    df2 = df1[["prop_"+str(i/n_actions) for i in range(n_actions)]]

    fig = plt.figure(figsize=(n_actions*2,n_thresholds))
    ax = sns.heatmap(df2, annot=True, fmt=".1g", yticklabels=[float('{:.2g}'.format(t)) for t in thresholds])
    ax.set_xlabel("Proposer Action Values")
    ax.set_ylabel("Threshold")
    plt.savefig("proposer_thresholds_vs_action_values.png")


    df2 = df1[["recv_"+str(i/n_actions) for i in range(n_actions)]]

    fig = plt.figure(figsize=(n_actions*2,n_thresholds))
    ax1 = sns.heatmap(df2, annot=True, fmt=".1g", yticklabels=[float('{:.2g}'.format(t)) for t in thresholds])
    ax1.set_xlabel("Receiver Action Values")
    ax1.set_ylabel("Threshold")
    plt.savefig("receiver_thresholds_vs_action_values.png")


    if (n_thresholds > 1):
        fig, axs = plt.subplots(n_thresholds, sharex=True, sharey=True)
        fig.suptitle('Rewards in time')
        for idx, t_value in enumerate(thresholds):
            df1 = df[df["threshold"] == t_value]
            df1 = df1[df1["train?"] == "train"]
            axs[idx].plot(df1["episode"], df1["rew_proposer"], label="rew_proposer")
            axs[idx].plot(df1["episode"], df1["rew_receiver"], label="rew_receiver")
            axs[idx].set_ylabel("Reward")
            axs[idx].grid()
        axs[n_thresholds-1].set_xlabel("Episode")
        plt.legend()
        plt.show()
    
    else:
        df1 = df1[df1["train?"] == "train"]
        fig, axs = plt.subplots(n_thresholds, sharex=True, sharey=True)
        fig.suptitle('Rewards in time')
        axs.plot(df1["episode"], df1["rew_proposer"], label="rew_proposer")
        axs.plot(df1["episode"], df1["rew_receiver"], label="rew_receiver")
        axs.set_ylabel("Reward")
        axs.grid()
        axs.set_xlabel("Episode")
        plt.legend()
        plt.show()


    df1 = df[df['train?'] == 'train']
    for i in range(n_actions):
        plt.plot(df1['episode'], df1["prop_"+str(i/n_actions)], label="prop_"+str(i/n_actions))
    plt.grid()
    plt.legend()
    plt.show()

    df1 = df[df['train?'] == 'train']
    for i in range(n_actions):
        plt.plot(df1['episode'], df1["recv_"+str(i/n_actions)], label="recv_"+str(i/n_actions))
    plt.grid()
    plt.legend()
    plt.show()