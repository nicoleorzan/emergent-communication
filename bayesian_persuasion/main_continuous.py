from agents import Sender, Rational_Receiver, Nature
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


params = {
    'prior': 0.7, # probability of being innocent
    'num_runs': 1,
    'num_probs': 7,
    'num_episodes': 1,
    'eval_episodes': 1,
    'epsilon0': 0.1,
    'eps_decay': 1E-4,
    'eps_min': 0.00001,
    "T": 0.5,
    'lr0': 0.001,
    'lr_decay': 0,
    'fair': True,
    'save_step': 40,
    'batch_size':1,
    'input_size':2,
    'seed':123
}


print("Fair?", params['fair'])
print(params['num_runs'], "runs and ", params['num_episodes'], "episodes")

torch.manual_seed(params['seed'])

def train(episode, send, recv, nature):

    truth_tmp = torch.Tensor([[1,0],[0,1]])
    truth_idxs = torch.Tensor([[0],[1]])
    #print("truth=", truth_tmp)
    probs, mex, logprobs = send.model.forward(truth_tmp.float())
    print("probs=", probs)
    print("mex=", mex)
    #print("logprobs=", logprobs)

    send.p_cond[0,0] = probs[0,0].T.detach()
    send.p_cond[1,0] = probs[0,1].T.detach()
    send.p_cond[0,1] = probs[1,0].T.detach()
    send.p_cond[1,1] = probs[1,1].T.detach()
    #print("send.p_cond=",send.p_cond)

    #truth, idx_truth = nature.get_state_of_world(batch_size)
    #print("truth=", truth)

    pred_r, _, _, _, _ = recv.predict(mex, send.p_cond)
    print("pred=", pred_r)
    
    if (params['fair'] == True):
        rew_s = nature.get_reward_fair(truth_idxs.float(), pred_r)
    else: 
        rew_s = nature.get_reward(truth_idxs.float(), pred_r)
    print("rew_s", rew_s)

    loss_s = send.loss(rew_s, logprobs)
    #print("loss=", loss_s)

    if send.model.opt is not None:

        send.model.opt.zero_grad()
        loss_s.backward()

        send.model.opt.step()

    return loss_s

    
def evaluation(params, send, recv, nature):

    """truth_tmp = torch.Tensor([[1,0],[0,1]]) #torch.Tensor([[0],[1]])
    probs, _, _ = send.model.forward(truth_tmp.float())
    send.p_cond = probs.detach().numpy().T

    truth, idx_truth = nature.get_state_of_world(batch_size)

    probs, mex, logprobs = send.model.forward(truth.float())

    pred_r, p_I_i, p_I_c, p_C_i, p_C_c = recv.predict(mex, send.p_cond)"""

    truth_tmp = torch.Tensor([[1,0],[0,1]])
    truth_idxs = torch.Tensor([[0],[1]])

    #print("truth=", truth_tmp)
    probs, mex, logprobs = send.model.forward(truth_tmp.float())
    
    send.p_cond[0,0] = probs[0,0].T.detach()
    send.p_cond[1,0] = probs[0,1].T.detach()
    send.p_cond[0,1] = probs[1,0].T.detach()
    send.p_cond[1,1] = probs[1,1].T.detach()
    #print("send.p_cond=",send.p_cond)

    pred_r, p_I_i, p_I_c, p_C_i, p_C_c = recv.predict(mex, send.p_cond)
    
    if (params['fair'] == True):
        rew_s = nature.get_reward_fair(truth_idxs.float(), pred_r)
    else: 
        rew_s = nature.get_reward(truth_idxs.float(), pred_r)

    eval_loss = send.loss(rew_s, logprobs)

    return eval_loss, rew_s, p_I_i, p_I_c, p_C_i, p_C_c



def main(params):

    i = 0

    num_episodes = params['num_episodes']

    df = pd.DataFrame(columns=['run', 'episode', 'p(i|I)', 'p(c|I)', 'p(i|C)', 'p(c|C)', 'p_I_i', 'p_I_c', 'p_C_i', 'p_C_c', 'loss', 'avg_rew'])

    for r in range(params['num_runs']):

        print("\nRun=", r)

        nature = Nature(params['prior'])
        send = Sender(params['input_size'], params['prior'], lr=params['lr0'])    
        recv = Rational_Receiver(prior_i = params['prior'])

        for ep in range(num_episodes):

            train_loss = train(params, send, recv, nature)

            ev_loss, rew_s, p_I_i, p_I_c, p_C_i, p_C_c  = evaluation(params, send, recv, nature)
            if (ep%30 == 0):
                p_cond = send.p_cond.detach().numpy()
                df.loc[i] = [r, ep, p_cond[0,0], p_cond[1,0], p_cond[0,1], p_cond[1,1], p_I_i, p_I_c, p_C_i, p_C_c , ev_loss.detach().numpy(), rew_s]#.mean()]
                i += 1

            if (ep%100 == 0):
                print("\nEpisode=", ep)
                print("Train loss=", train_loss)
                print("Evaluation loss=", ev_loss)

        
    if (num_episodes > 1):
        if (params['fair'] == True):
            df.to_csv('dataset2_'+str(num_episodes)+'_fair_locale.csv', index = False)
        else: 
            df.to_csv('dataset2_'+str(num_episodes)+'_unfair_locale.csv', index = False)

#print("Final values: p(i|I) = ", send.p_cond[0,0])
#print("p(i|C)=", send.p_cond[0,1])
#print("p(c|I)=", send.p_cond[1,0])
#print("p(c|C)=", send.p_cond[1,1])



if __name__ == "__main__":
    main(params = params)