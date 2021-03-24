from agents.DQNAgent import DQNAgent
from utils.Environments import DiscEnv
import matplotlib.pyplot as plt
import numpy as np

num_act = 3
input_dim  = 1
sender = DQNAgent(input_dim=input_dim, num_actions=num_act, batch_size=64, lr=0.01, eps_start=0.9)
receiver = DQNAgent(input_dim=input_dim, num_actions=num_act, batch_size=64, lr=0.01, eps_start=0.9)
env = DiscEnv(num_obs=num_act, num_actions=num_act)

num_episodes = 40000
for ep in range(num_episodes):
    if (ep%1000 == 0):
        print("episode=", ep)
        
    obs = env.get_observation()
    mex = sender.take_action(obs)
    act = receiver.take_action(mex)
    rew = env.get_reward(obs, act)
    
    sender.store_transition(obs, mex, rew)
    receiver.store_transition(mex, act, rew)
    
    sender.train(ep)
    receiver.train(ep)
    
print(sender.epsilon)

for s in range(num_act):
    print(sender.get_action_probabilities(s))

for s in range(num_act):
    plt.plot(np.linspace(0, num_act, num_act), sender.get_action_probabilities(s), label=str(s))
plt.legend()
plt.show()