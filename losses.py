import torch
import torch.nn as nn
import torch.nn.functional as F


def macro_dice_loss(probas, labels, smooth=1):

    probas = F.softmax(probas, dim=1)
    C = probas.size(1)
    
    losses_b = []
    for bs in range(probas.shape[0]):
        losses = []
        for c in range(C):
            fg = (labels[bs] == c).float()[0]
            if fg.sum() == 0:
                continue
            class_pred = probas[bs, c]
            p0 = class_pred
            g0 = fg
            numerator = 2 * torch.sum(p0 * g0) + smooth
            denominator = torch.sum(p0) + torch.sum(g0) + smooth
            # print(bs, c, p0.shape, g0.shape, numerator, denominator)
            losses.append(1 - ((numerator) / (denominator)))
        losses_b.append(sum(losses) / len(losses))
    return sum(losses_b) / len(losses_b)


def topk_cross_entropy_loss(probas, labels, k=1):
    
    b, _, h, w = probas.shape
    losses_b = []
    for bs in range(b):
        ce = nn.CrossEntropyLoss(reduction="none")(probas[bs:bs+1], labels[bs:bs+1,0].long())
        top_k = int(h * w * k)
        ce = ce.view(1, h*w)
        top_k_loss, _ = torch.topk(ce, k=top_k, dim=1)
        # print(top_k_loss.shape)
        top_k_loss = top_k_loss.mean()
        losses_b.append(top_k_loss)
    return sum(losses_b) / len(losses_b)
