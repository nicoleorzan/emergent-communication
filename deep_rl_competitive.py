from agents.DQNAgent_improved import DQNAgent_improved
from utils.Environments import DiscEnv
import matplotlib.pyplot as plt
import numpy as np

"""
STILL NOT WORKING
"""

num_act = 3
input_dim  = 1
batch_size = 100
penalty = 1
returns1 = []
returns2 = []

sender1 = DQNAgent_improved(input_dim=input_dim, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.9, intermed_nodes=num_act, 
eps_min=0.01, eps_dec=5e-5, capacity=5000)
sender2 = DQNAgent_improved(input_dim=input_dim, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.9, intermed_nodes=num_act, 
eps_min=0.01, eps_dec=5e-5, capacity=5000)

receiver1 = DQNAgent_improved(input_dim=2, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.9, intermed_nodes=num_act, eps_min=0.01, 
eps_dec=5e-5, capacity=5000)
receiver2 = DQNAgent_improved(input_dim=2, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.9, intermed_nodes=num_act, eps_min=0.01, 
eps_dec=5e-5, capacity=5000)

env = DiscEnv(num_obs=num_act, num_actions=num_act)

num_episodes = 101

for ep in range(num_episodes):
    if (ep%1000 == 0):
        print("episode=", ep)
        print("epsilon", sender1.epsilon)

    obs1 = env.get_observation()
    obs2 = env.get_observation()

    s_s1 = sender1.take_action(obs1)
    s_s2 = sender2.take_action(obs2)

    mex1b = np.array((s_s1, s_s2))
    mex2b = np.array((s_s2, s_s1))
    print("mex1b=", mex1b)
    print("mex2b=", mex2b)
    act_r1 = receiver1.take_action(mex1b)
    print("act1=", act_r1)

    act_r2 = receiver2.take_action(mex2b)
    print("act2=", act_r2)
    rew1 = env.get_reward(obs1, act_r1)
    rew2 = env.get_reward(obs2, act_r2)

    if (rew1 != rew2):
        if (rew1 > rew2):
            rew1 = rew1*2
            rew2 = rew2 - penalty
        else:
            rew2 = rew2*2
            rew1 = rew1 - penalty

    returns1.append(rew1)
    returns2.append(rew2)
    print("store")
    sender1.store_transition(obs1, s_s1, rew1, None)
    sender2.store_transition(obs2, s_s2, rew2, None)
    print("store receiver")
    #receiver1.store_transition(mex1b, act_r1, rew1, None)
    #receiver2.store_transition(mex2b, act_r2, rew2, None)
    
    print("sender1 train")
    sender1.train(ep)
    print("receiver1 train")
    receiver1.train(ep)
    sender2.train(ep)
    print("receiver2 train")
    receiver2.train(ep)
    
print("epsilon", sender1.epsilon)
print("sender action probabilities")
for s in range(num_act):
    print(sender1.get_action_probabilities(s))

print("receiver action probabilities")
for s in range(num_act):
    print(receiver1.get_action_probabilities(np.array((s,s))))

for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), sender1.get_action_probabilities(s), label="sender1")
    plt.plot(np.linspace(0, num_act, num_act), sender2.get_action_probabilities(s), label="sender2")
plt.legend()
plt.savefig("senders_policy_comp.png")
plt.show()

"""for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), receiver1.get_action_probabilities(np.array((s,s))), label=str(s))
plt.legend()
plt.savefig("receiver1_policy_comp.png")
"""


#test phase:
test_episodes = 500
sender1.epsilon = 0
receiver1.epsilon = 0
sender2.epsilon = 0
receiver2.epsilon = 0

for ep in range(test_episodes):

    obs1 = env.get_observation()
    obs2 = env.get_observation()

    mex1 = sender1.take_action(obs1)
    mex2 = sender2.take_action(obs2)

    mex1b = np.array((mex1, mex2))
    mex2b = [mex2]
    mex2b.append(mex1)
    mex2b = np.array((mex2, mex1))
    #print("mex1b=", mex1b)
    #print("mex2b=", mex2b)
    act1 = receiver1.take_action(mex1b)
    #print("act1=", act1)

    act2 = receiver2.take_action(mex2b)
    #print("act2=", act2)
    rew1 = env.get_reward(obs1, act1)
    rew2 = env.get_reward(obs2, act2)

    if (rew1 != rew2):
        if (rew1 > rew2):
            rew1 = rew1*2
            rew2 = rew2 - penalty
        else:
            rew2 = rew2*2
            rew1 = rew1 - penalty
    returns1.append(rew1)
    returns2.append(rew2)


plt.plot(np.linspace(0, len(returns1), len(returns1)), returns1, label="returns1")
plt.plot(np.linspace(0, len(returns2), len(returns2)), returns2, label="returns2")
plt.legend()
plt.savefig("returns_competitive.png")