'''
This is our method core code for selection and learning,total code will be open after accepted.Thanks
'''
import torch
import numpy as np
import torch.nn as nn
from sklearn.mixture import GaussianMixture

CE = nn.CrossEntropyLoss(reduction='none')
#Enhanced-selection
def two_view_select(model,u,eval_loader,all_loss,r):
    model.eval()
    pre = np.ones(50000)
    pre_loss = np.ones(50000)
    input_u = torch.abs(u.detach()).cpu().numpy().squeeze(1)
    input_u = input_u.reshape(-1, 1)
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index,_) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            resdict = model(inputs, mode='eval')
            outputs = resdict['output']
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    gmm_u = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
    gmm_u.fit(input_u)
    prob_u = gmm_u.predict_proba(input_u)
    prob_u = prob_u[:, gmm_u.means_.argmin()]

    gmm_loss = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
    gmm_loss.fit(input_loss)
    prob_loss = gmm_loss.predict_proba(input_loss)
    prob_loss = prob_loss[:, gmm_loss.means_.argmin()]
    pre[prob_u >= 0.5] = 0
    print(f"u num clean:{np.sum(pre == 0)}")
    print(f"u num noise:{np.sum(pre == 1)}")
    pre_loss[prob_loss >= 0.5] = 0
    print(f"loss num clean:{np.sum(pre_loss == 0)}")
    print(f"loss num noise:{np.sum(pre_loss == 1)}")
    return pre,prob_u,pre_loss,all_loss
