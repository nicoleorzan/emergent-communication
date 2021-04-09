from agents.DQNAgent import DQNAgent
from utils.Environments import DiscEnv
import matplotlib.pyplot as plt
import numpy as np

num_act = 3
input_dim  = 1
batch_size = 100
sender = DQNAgent(input_dim=input_dim, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.99, intermed_nodes=num_act, 
eps_min=0.01, eps_dec=5e-5, capacity=7000) #5
receiver = DQNAgent(input_dim=input_dim, num_actions=num_act, batch_size=batch_size, lr=0.01, eps_start=0.99, intermed_nodes=num_act,
eps_min=0.01, eps_dec=5e-5, capacity=7000)
env = DiscEnv(num_obs=num_act, num_actions=num_act)
returns = []

print("sender action probabilities")
for s in range(num_act):
    print(sender.get_action_probabilities(s))

print("receiver action probabilities")
for s in range(num_act):
    print(receiver.get_action_probabilities(s))

num_episodes = 20000

for ep in range(num_episodes):
    if (ep%1000 == 0):
        print("episode=", ep)
        print("epsilon", sender.epsilon)
        
    obs = env.get_observation()
    mex = sender.take_action(obs)
    act = receiver.take_action(mex)
    rew = env.get_reward(obs, act)
    
    sender.store_transition(obs, mex, rew)
    receiver.store_transition(mex, act, rew)
    returns.append(rew)
    
    sender.train(ep)
    receiver.train(ep)
    
print("epsilon", sender.epsilon)
print("sender action probabilities")
for s in range(num_act):
    print(sender.get_action_probabilities(s))

print("receiver action probabilities")
for s in range(num_act):
    print(receiver.get_action_probabilities(s))

for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), sender.get_action_probabilities(s), label=str(s))
plt.legend()
plt.savefig("sender_policy"+str(num_act)+"_dqn_discrete.png")
plt.show()

for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), receiver.get_action_probabilities(s), label=str(s))
plt.legend()
plt.savefig("receiver_policy"+str(num_act)+"_dqn_discrete.png")
plt.show()

#test phase:
test_episodes = 3000
sender.epsilon = 0
receiver.epsilon = 0
for ep in range(test_episodes):

    obs = env.get_observation()
    mex = sender.take_action(obs)
    act = receiver.take_action(mex)
    rew = env.get_reward(obs, act)
    returns.append(rew)

plt.plot(np.linspace(0, len(returns), len(returns)), returns, label="returns")
plt.legend()
plt.savefig("returns_dqn_discrete.png")
plt.show()
