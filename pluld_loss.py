import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.registry import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from torch.nn import _reduction as _Reduction

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def pluld_loss(pred, #cls_score
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         pos_weight=None,
                         #size_average=None,
                         #reduce=None,
                         lamb = 0.9):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    # # first:-(ylog(p)+(1-y)log(1-p))
    # label=label.float()
    # p = torch.tensor([1]).cuda(0)
    # sig_x = torch.sigmoid(pred)
    # log_sig_x = torch.log(sig_x)
    # sub_1_x = torch.sub(p,sig_x)
    # sub_1_y = torch.sub(p,label)
    # log_1_x = torch.log(sub_1_x)
    # if pos_weight is None:
    #     loss = -(torch.add(torch.mul(label, log_sig_x), torch.mul(sub_1_y, log_1_x)))
    # else:
    #     loss = -(torch.add(torch.mul(torch.mul(label, log_sig_x), pos_weight), torch.mul(sub_1_y,log_1_x)))
    # loss = weight_reduce_loss(loss,weight, reduction=reduction, avg_factor=avg_factor)
    #
    # return loss

    # #second: only lamb
    label = label.float()
    p = torch.tensor([1]).cuda(0)
    sig_x = torch.sigmoid(pred)
    log_sig_x = torch.log(sig_x)
    sub_1_x = torch.sub(p, sig_x)
    sub_1_y = torch.sub(p, label)
    log_1_x = torch.log(sub_1_x)
    if pos_weight is None:
        loss = -(torch.add(torch.mul(label, log_sig_x), torch.mul(torch.mul(sub_1_y, log_1_x),lamb)))
    else:
        loss = -(torch.add(torch.mul(torch.mul(label, log_sig_x), pos_weight), torch.mul(torch.mul(sub_1_y, log_1_x),lamb)))
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


    # #third: only mask
    # label=label.float()
    # p = torch.tensor([1]).cuda(0)
    # sig_x = torch.sigmoid(pred)
    # log_sig_x = torch.log(sig_x) #log(sigmoid(pred))
    # sub_1_x = torch.sub(p,sig_x) #1-sigmoid(pred)
    # sub_1_y = torch.sub(p,label) #label_neg=1-label
    # mask_sub_1_y = get_mask(sub_1_y) #mask_label_neg
    # log_1_x = torch.log(sub_1_x) #log(1-sigmoid(pred))
    # if pos_weight is None:
    #     loss = -(torch.add(torch.mul(label, log_sig_x), torch.mul(mask_sub_1_y, log_1_x)))
    # else:
    #     loss = -(torch.add(torch.mul(torch.mul(label, log_sig_x), pos_weight), torch.mul(mask_sub_1_y,log_1_x)))
    # loss = weight_reduce_loss(loss,weight, reduction=reduction, avg_factor=avg_factor)
    # return loss

    # #forth: mask + lamb
    # label=label.float()
    # p = torch.tensor([1]).cuda(0)
    # sig_x = torch.sigmoid(pred)
    # log_sig_x = torch.log(sig_x) #log(sigmoid(pred))
    # sub_1_x = torch.sub(p,sig_x) #1-sigmoid(pred)
    # sub_1_y = torch.sub(p,label) #label_neg=1-label
    # mask_sub_1_y = get_mask(sub_1_y) #mask_label_neg
    # log_1_x = torch.log(sub_1_x) #log(1-sigmoid(pred))
    # if pos_weight is None:
    #     loss = -(torch.add(torch.mul(label, log_sig_x), torch.mul(torch.mul(mask_sub_1_y, log_1_x),lamb)))
    # else:
    #     loss = -(torch.add(torch.mul(torch.mul(label, log_sig_x), pos_weight), torch.mul(torch.mul(mask_sub_1_y, log_1_x),lamb)))
    # loss = weight_reduce_loss(loss,weight, reduction=reduction, avg_factor=avg_factor)
    # return loss

#generate mask to reduce label_neg
def get_mask(label):
    row = label.numel()
    x = torch.randn(row,1)
    mask = x > 0
    mask = mask.type(torch.uint8).cuda(0)
    label = label*mask
    return label

@LOSSES.register_module
class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(WeightedCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid:
            self.cls_criterion = pluld_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,

                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction) #总损失除以样本数
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
