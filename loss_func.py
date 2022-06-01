import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F


# def softmax_focal_loss(x, target, gamma=2., alpha=0.25):
#     n = x.shape[0]
#     device = target.device
#     range_n = torch.arange(0, n, dtype=torch.int64, device=device)

#     pos_num =  float(x.shape[1])
#     p = torch.softmax(x, dim=1)
#     p = p[range_n, target]
#     loss = -(1-p)**gamma*alpha*torch.log(p)
#     return torch.sum(loss) / pos_num


# class FocalLoss(nn.Module):
#     """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
#     It is essentially an enhancement to cross entropy loss and is
#     useful for classification tasks when there is a large class imbalance.
#     x is expected to contain raw, unnormalized scores for each class.
#     y is expected to contain class labels.
#     Shape:
#         - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
#         - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
#     """

#     def __init__(self,
#                  alpha: Optional[Tensor] = None,
#                  gamma: float = 0.,
#                  reduction: str = 'mean',
#                  ignore_index: int = -100):
#         """Constructor.
#         Args:
#             alpha (Tensor, optional): Weights for each class. Defaults to None.
#             gamma (float, optional): A constant, as described in the paper.
#                 Defaults to 0.
#             reduction (str, optional): 'mean', 'sum' or 'none'.
#                 Defaults to 'mean'.
#             ignore_index (int, optional): class label to ignore.
#                 Defaults to -100.
#         """
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(
#                 'Reduction must be one of: "mean", "sum", "none".')

#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#         self.nll_loss = nn.NLLLoss(
#             weight=alpha, reduction='none', ignore_index=ignore_index)

#     def __repr__(self):
#         arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
#         arg_vals = [self.__dict__[k] for k in arg_keys]
#         arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
#         arg_str = ', '.join(arg_strs)
#         return f'{type(self).__name__}({arg_str})'

#     def forward(self, x: Tensor, y: Tensor) -> Tensor:
#         if x.ndim > 2:
#             # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
#             c = x.shape[1]
#             x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
#             # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
#             y = y.view(-1)

#         unignored_mask = y != self.ignore_index
#         y = y[unignored_mask]
#         if len(y) == 0:
#             return 0.
#         x = x[unignored_mask]

#         # compute weighted cross entropy term: -alpha * log(pt)
#         # (alpha is already part of self.nll_loss)
#         log_p = F.log_softmax(x, dim=-1)
#         ce = self.nll_loss(log_p, y)

#         # get true class column from each row
#         all_rows = torch.arange(len(x))
#         log_pt = log_p[all_rows, y]

#         # compute focal term: (1 - pt)^gamma
#         pt = log_pt.exp()
#         focal_term = (1 - pt)**self.gamma

#         # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
#         loss = focal_term * ce

#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()

#         return loss


# def focal_loss(alpha: Optional[Sequence] = None,
#                gamma: float = 0.,
#                reduction: str = 'mean',
#                ignore_index: int = -100,
#                device='cpu',
#                dtype=torch.float32) -> FocalLoss:
#     """Factory function for FocalLoss.
#     Args:
#         alpha (Sequence, optional): Weights for each class. Will be converted
#             to a Tensor if not None. Defaults to None.
#         gamma (float, optional): A constant, as described in the paper.
#             Defaults to 0.
#         reduction (str, optional): 'mean', 'sum' or 'none'.
#             Defaults to 'mean'.
#         ignore_index (int, optional): class label to ignore.
#             Defaults to -100.
#         device (str, optional): Device to move alpha to. Defaults to 'cpu'.
#         dtype (torch.dtype, optional): dtype to cast alpha to.
#             Defaults to torch.float32.
#     Returns:
#         A FocalLoss object
#     """
#     if alpha is not None:
#         if not isinstance(alpha, Tensor):
#             alpha = torch.tensor(alpha)
#         alpha = alpha.to(device=device, dtype=dtype)

#     fl = FocalLoss(
#         alpha=alpha,
#         gamma=gamma,
#         reduction=reduction,
#         ignore_index=ignore_index)
#     return fl


# # x, y = torch.randn(1, 17), (torch.rand(1) > .5).long()

# # print(softmax_focal_loss(x, y))

# # print(focal_loss()(x, y))

# class FocalLoss1(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight=None, gamma=2,reduction='mean'):
#         super(FocalLoss1, self).__init__(weight,reduction=reduction)
#         self.gamma = gamma
#         self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

#     def forward(self, input, target):

#         ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
#         return focal_loss
    
# def FocalLoss2(outputs, targets, alpha=0.25, gamma=2):
#     ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
#     pt = torch.exp(-ce_loss)
#     focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch
#     return focal_loss

class FocalLoss3(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss3, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, weights, gamma=2):
#         super().__init__()
#         self.weights = weights
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         inputs = inputs.squeeze()
#         targets = targets.squeeze()

#         BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.weights[targets]*(1-pt)**self.gamma * BCE_loss

#         return F_loss.mean()