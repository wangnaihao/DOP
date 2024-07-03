from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
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
            loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)
        else:
            prediction_l = y_l.detach()
            loss_l = self.eta_prime * (-torch.mean(torch.sum(F.log_softmax(y_l_adv, dim=1) * prediction_l, dim=1)))
            prediction_u = y_u.detach()
            loss_u = -torch.mean(torch.sum(F.log_softmax((1. - F.softmax(y_u_adv, dim=1)), dim=1) * prediction_u, dim=1))
        return loss_l + loss_u
def linear_rampup2(current, rampup_length,start):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current - start >= rampup_length:
        return 1.0
    else:
        return (current - start) / rampup_length
