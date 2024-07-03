import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with log_softmax, which is more numerically stable."""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y, labels, reduction='mean'):
        log_p = torch.log_softmax(y, dim=1)
        return F.nll_loss(log_p, labels, reduction=reduction)

class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """Self-training loss with confidence threshold."""

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.criterion = CrossEntropyLoss()

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (self.criterion(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels


def shift_log(x, offset=1e-6):
    return torch.log(torch.clamp(x + offset, max=1.))

class WorstCaseEstimationLoss(nn.Module):
    def __init__(self, eta_prime = 0.9):
        super(WorstCaseEstimationLoss, self).__init__()
        self.eta_prime = eta_prime

    def forward(self, y_l, y_l_adv, y_u, y_u_adv,mix=False):
        if not mix:
            _, prediction_l = y_l.max(dim=1)
            loss_l = self.eta_prime * F.cross_entropy(y_l_adv, prediction_l)

            _, prediction_u = y_u.max(dim=1)
            loss_u = F.nll_loss((shift_log(1. - F.softmax(y_u_adv, dim=1))), prediction_u)
        else:
            prediction_l = y_l.detach()
            loss_l = self.eta_prime * (-torch.mean(torch.sum(F.log_softmax(y_l_adv, dim=1) * prediction_l, dim=1)))
            prediction_u = y_u.detach()
            loss_u = -torch.mean(torch.sum(F.log_softmax(y_u_adv, dim=1) * prediction_u, dim=1))
        return loss_l + loss_u
class newWorstCaseEstimationLoss(nn.Module):
    def __init__(self, eta_prime = 0.9):
        super(newWorstCaseEstimationLoss, self).__init__()
        self.eta_prime = eta_prime
        self.gce_loss = GCELoss
    def forward(self, y_c, y_c_adv, y_h, y_h_adv,y_n,y_n_adv):
        loss = 0.
        if len(y_c) > 0:
            prediction_l = y_c.detach()
            loss_l = self.eta_prime * (-torch.mean(torch.sum(F.log_softmax(y_c_adv, dim=1) * prediction_l, dim=1)))
            loss += loss_l
        if len(y_h) > 0:
            pred = F.softmax(y_h_adv, dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            prediction_h = y_h.detach()
            loss_h = self.eta_prime * ((1. - torch.pow(torch.sum(prediction_h * pred, dim=1), 0.7)) / 0.7).mean()
            loss += loss_h
        if len(y_n) > 0:
            prediction_u = y_n.detach()
            loss_u = -torch.mean(torch.sum(F.log_softmax(y_n_adv, dim=1) * prediction_u, dim=1))
            loss += loss_u

        else:
            prediction_l = y_l.detach()
            loss_l = self.eta_prime * (-torch.mean(torch.sum(F.log_softmax(y_l_adv, dim=1) * prediction_l, dim=1)))
            prediction_u = y_u.detach()
            y_u_adv_flip = shift_log(1 - F.softmax(y_u_adv,dim = 1))
            loss_u = -torch.mean(torch.sum(F.softmax(y_u_adv_flip,dim = 1) * prediction_u, dim=1))
        return loss
