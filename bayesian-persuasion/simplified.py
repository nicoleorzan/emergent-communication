from agents_simplified import Sender_Mini
import pandas
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def get_reward_and_loss(truth, pred): # works. Not using REINFORCE here
    return torch.mean(torch.abs(truth - pred.squeeze()))

def get_reward_only(truth, sampled):
    rew = -torch.abs(truth - sampled.squeeze()).detach()
    return rew

np.random.seed(123)
lr = 0.02 # 0.01, 0.02 oppure 0.03 con drop rate 0.1
drop_rate = 0.0
batch_size = 512*2
write = False
prior = 0.7 # probability of being innocent
probbs = [prior, 1-prior]
num_episodes = 1000

print("Parameters: lr=", lr, ", drop_rate=", drop_rate)

df = pandas.DataFrame(columns=['episode', 'mean_i', 'mean_c', 'var_i', 'var_c', 'loss'])

send = Sender_Mini(prior, lr=lr, drop_rate=drop_rate)
truth_prob = np.array([prior, 1-prior])


if (write == True):
    writer = SummaryWriter()
    writer.add_graph(send.model, torch.Tensor([1, 0]))

for ep in range(num_episodes):

    idxs = np.random.choice([0,1], batch_size, p=[0.5, 0.5]) # l'assimmetria e` un problema per la NN!
    idxs1 = [0 for i in range(int(batch_size/2))]
    idxs2 = [1 for i in range(int(batch_size/2))]
    idxs = idxs1 + idxs2
    indices = torch.Tensor(idxs).to(torch.int64)
    truth_batch = torch.nn.functional.one_hot(indices, 2).float()
    truth_values = torch.Tensor([probbs[idx] for idx in idxs])
    pred, var, sampled, logprobs = send.model.forward(truth_batch)
    
    rew = get_reward_only(truth_values, pred) # kind of works
    loss = send.lossfunc(rew, logprobs)
    
    #loss = get_reward_and_loss(truth_values, sampled) # works
    #print(truth_batch == truth_values)

    #print("truth batch=", truth_batch, "truth values=", truth_values, "\npred=", pred, "\nlogprobs=", logprobs, "\nrew=", rew, "\nloss=", loss, "\n")

    if (ep%100 == 0):
        print("\nEpisode =", ep)
        print("loss=", loss)

    if send.model.opt is not None:
        send.model.opt.zero_grad()
        loss.backward()

        send.model.opt.step()

    # TEST

    mean_i, var_i, sampled_i, logprobs_i = send.model.forward(torch.Tensor([1,0]))

    mean_c, var_c, sampled_c, logprobs_c = send.model.forward(torch.Tensor([0,1]))

    df.loc[ep] = [ep, mean_i.detach().numpy()[0], mean_c.detach().numpy()[0], var_i.detach().numpy()[0], var_c.detach().numpy()[0], loss.detach().numpy()]
    
    if (write == True):
        writer.add_scalar("loss/train", loss, ep)
        writer.add_scalar("lr/train", lr, ep)
        writer.add_scalar("drop/train", send.model.drop_rate, ep)
        writer.add_scalar("mean_i/train", mean_i, ep)
        writer.add_scalar("mean_c/train", mean_c, ep)


if (write == True):
    writer.close()

df.to_csv(r'dataset_simplified.csv', index = False)

plt.plot(df['episode'],df['mean_i'], label='mean_i')
plt.plot(df['episode'],df['mean_c'], label='mean_c')
plt.legend()
plt.grid()
plt.show()

plt.plot(df['episode'],df['loss'], label='mean_c')
plt.grid()
plt.show()