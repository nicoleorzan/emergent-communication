from agents.DQNAgent_improved import DQNAgent_improved
from utils.Environments import DiscEnv
import matplotlib.pyplot as plt
import numpy as np
import random

num_act = 3
input_dim  = 1
batch_size = 100
sender = DQNAgent_improved(input_dim=input_dim, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.99,
 intermed_nodes = num_act, eps_min=0.01, eps_dec=5e-5, capacity=8000)
receiver = DQNAgent_improved(input_dim=input_dim, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.99, 
 intermed_nodes = num_act, eps_min=0.01, eps_dec=5e-5, capacity=8000)
env = DiscEnv(num_obs=num_act, num_actions=num_act)

print("sender action probabilities")
for s in range(num_act):
    print(sender.get_action_probabilities(s))

print("receiver action probabilities")
for s in range(num_act):
    print(receiver.get_action_probabilities(s))

print("Learning:")

num_episodes = 20000

vals = [i for i in range(num_act)]
for ep in range(num_episodes):
    if (ep%1000 == 0):
        print("epsilon=", sender.epsilon)
        print("episode=", ep)
        
    obs = env.get_observation()

    mex = sender.take_action(obs)
    act = receiver.take_action(mex)
    rew = env.get_reward(obs, act)

    s1 = random.choice(vals)

    sender.store_transition(obs, mex, rew, None)
    receiver.store_transition(mex, act, rew, None)
    
    sender.train(ep)
    receiver.train(ep)

print(sender.epsilon)
print("sender action probabilities")
for s in range(num_act):
    print(sender.get_action_probabilities(s))

print("receiver action probabilities")
for s in range(num_act):
    print(receiver.get_action_probabilities(s))

for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), sender.get_action_probabilities(s), label=str(s))
plt.legend()
plt.savefig("sender_policy"+str(num_act)+"_improved_dqn_discrete.png")
plt.show()

for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), receiver.get_action_probabilities(s), label=str(s))
plt.legend()
plt.savefig("receiver_policy"+str(num_act)+"_improved_dqn_discrete.png")
plt.show()
