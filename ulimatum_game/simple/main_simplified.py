import matplotlib.pyplot as plt
from simpleagents import Sender, Receiver
import pandas as pd
import seaborn as sns

n_actions = 10
epsilon = 0.2
lr = 0.2
n_thresholds = 6
thresholds = [(i/n_thresholds) for i in range(n_thresholds)]

num_episodes = 10000
ep_step = 10


def get_reward(resp, p):
    if (resp == 1):
        return 1.-p
    else: 
        return 0.


def train(run_n, threshold, n_episodes, df):
    proposer = Sender(idx = 0, lr = lr, n_actions=n_actions, epsilon=epsilon)
    receiver = Receiver(idx = 1, lr = lr, threshold=threshold, n_states=n_actions, epsilon=epsilon)

    for ep in range(1,n_episodes+1):
        
        prop_idx = proposer.epsilon_greedy_proposal()
        prop = proposer.act(prop_idx)
        resp = receiver.threshold_recv(prop)

        rew = get_reward(resp, prop)
        proposer.TD_prop(prop_idx, rew)

        if (ep%ep_step == 0):
            df.loc[int(run_n*num_episodes/ep_step+ep/ep_step)] = [run_n, ep, threshold, epsilon, rew] + list(proposer.q_prop)


def main(n_episodes):

    df = pd.DataFrame(columns = ['run', 'episode', 'threshold', 'epsilon', 'rew'] + [str(i/n_actions) for i in range(n_actions)])

    for run, threshold in enumerate(thresholds):
        print("Run number=", run, " with threshold=", threshold)
        train(run, threshold, n_episodes, df)

    return df



if __name__ == "__main__":

    df = main(n_episodes = num_episodes)

    df1 = df[df["episode"] == df.episode.max()]
    df1 = df1[[str(i/n_actions) for i in range(n_actions)]]
    ax = sns.heatmap(df1, annot=True, fmt=".1g", yticklabels=[float('{:.2g}'.format(i/n_thresholds)) for i in range(n_thresholds)])
    ax.set_xlabel("Proposer Action Values")
    ax.set_ylabel("Threshold")
    plt.savefig("Thresholds_vs_action_values.png")
    #plt.show()


# interesting ideas:
# ToM
# NN as proposer?
# receiver that learns too