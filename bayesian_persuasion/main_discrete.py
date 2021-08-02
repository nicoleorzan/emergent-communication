from agents import DiscreteSender, Rational_Receiver, Nature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import string
import os
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


params = {
    'prior': 0.7, # probability of being innocent
    'num_runs': 1,
    'num_probs': 7,
    'num_episodes': 200000,
    'eval_episodes': 1000,
    'epsilon0': 0.1,
    'eps_decay': 1E-4,
    'eps_min': 0.00001,
    "T": 0.5,
    'lr0': 0.5,
    'lr_decay': 1E-4,
    'fair': False,
    'save_step': 40
}


print("Is reward fair?", params['fair'])
print("Discrete sender experiment with ", params['num_probs'], "probabilities,")
print(params['num_runs'], "runs and ", params['num_episodes'], "episodes")


def train(params, send, recv, nature):

    p_i_I_idx, p_i_C_idx = send.choose_signal()

    _, idx_truth = nature.get_state_of_world(batch_size=1)
    idx_truth = idx_truth.numpy()
    
    mex = send.sample_mex(idx_truth)

    pred_r, p_I_i, p_I_c, p_C_i, p_C_c = recv.predict(mex, send.p_cond)

    if (params['fair'] == True):
        rew_s = nature.get_reward_fair_discrete(idx_truth, pred_r)
    else: 
        rew_s = nature.get_reward(idx_truth, pred_r)
                
    send.update(p_i_I_idx, p_i_C_idx, rew_s)

    send.rewards[p_i_I_idx, p_i_C_idx] += rew_s

    return p_i_I_idx, p_i_C_idx, p_I_i, p_I_c, p_C_i, p_C_c, rew_s



def main(params):

    dirname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    path = os.path.join('.', dirname)
    os.mkdir(path)

    with open(dirname+'/params.json', 'w') as fp:
        json.dump(params, fp)
    print("Created experiment directory "+dirname)

    num_episodes = params['num_episodes']
    decay_rate = params['lr_decay']
    num_probs = params['num_probs']

    df = pd.DataFrame(columns=['run', 'episode', 'epsilon', 'lr', 'i', 'p(i|I)', 'p(c|I)', 'p(i|C)', 'p(c|C)', 'p_I_i', 'p_I_c', 'p_C_i', 'p_C_c', 'rew'] + ["q_0"+str(i) for i in range(num_probs+1)] + ["q_1"+str(i) for i in range(num_probs+1)])

    i = 0

    print("\n==> Start Training")

    for r in range(params['num_runs']):

        nature = Nature(params['prior'])
        send = DiscreteSender(params = params)    
        recv = Rational_Receiver(prior_i = params['prior'])
        
        with open(dirname+'/q_mat.txt', 'w') as f:
            f.write(str(send.q))

        for ep in range(num_episodes):
            
            if (ep%5000 == 0 and send.epsilon > 0.):
                print("\nEpisode:", ep)
                send.epsilon -= send.eps_decay if send.epsilon > send.eps_min else send.eps_min
                send.lr *= (1. / (1. + decay_rate * ep))
                print("send.epsilon=", send.epsilon)
                print("send.lr=", send.lr)

            p_i_I_idx, p_i_C_idx, p_I_i, p_I_c, p_C_i, p_C_c, rew_s = train(params, send, recv, nature)

            if (ep%params['save_step'] == 0):
                
                df.loc[i] = [r, ep, send.epsilon, send.lr, i, send.p_cond[0,0], send.p_cond[1,0], send.p_cond[0,1], send.p_cond[1,1], p_I_i, p_I_c, p_C_i, p_C_c, rew_s] + [send.q[0,i] for i in range(num_probs+1)] + [send.q[1,i] for i in range(num_probs+1)]
                i = i+1

            with open(dirname+'/q_mat.txt', 'a') as f:
                f.write(str(send.q))

            if (ep == num_episodes - 1):
                print("\n==> End")
                print("send.q=", send.q, "\n")
                print("p_i_I_idx", p_i_I_idx, ", p_i_C_idx", p_i_C_idx)
                print("q_max=", send.q[p_i_I_idx, p_i_C_idx])
                print("send.p_cond=", send.p_cond)
                print("\nrewards=",send.rewards)


    df.to_csv(dirname+'/dataset_'+str(num_episodes)+'_'+str(num_probs)+'_discrete_locale.csv', index = False)
    

    yticklabels = [i for i in range(num_probs)]

    ax = sns.heatmap(send.q, yticklabels=yticklabels)
    ax.set_title("Q Values ")
    plt.savefig(dirname+"/Qmat_"+str(num_probs)+"_episodes"+str(num_episodes)+'.png', bbox_inches='tight')
    plt.show()

    ax = sns.heatmap(send.rewards/send.q_counts, yticklabels=yticklabels)
    ax.set_title("Rewards ")
    plt.savefig(dirname+"/Reward_"+str(num_probs)+"_episodes"+str(num_episodes)+'.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main(params=params)