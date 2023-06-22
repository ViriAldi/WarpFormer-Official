import torch
import torch.nn as nn

def load_network(net, pretrained_dir):
    pretrained = torch.load(pretrained_dir)
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del (pretrained)
    return net

def load_old_arch(model: nn.Module, state_dict_old):
    state_dict = {}
    for k, v in state_dict_old.items():
        if k.startswith("unet."):
            k = k[5:]

        if k.startswith("id_bank12."):
            k = k.replace("id_bank12.", "id_bank.")
        elif k.startswith("id_bank02."):
            k = k.replace("id_bank02.", "id_bank.")
        elif k.startswith("cross_attn12."):
            k = k.replace("cross_attn12.", "cross_attn_short_term.")
        state_dict[k] = v
    missing_keys = set()
    for k in model.state_dict():
        if k.startswith("flow_estimator"):
            continue
        elif k not in state_dict:
            missing_keys.add(k)
    model.load_state_dict(state_dict, strict=False)
    return missing_keys

def count_params(model: nn.Module, only_trainable=False):
    return sum(p.numel() for p in model.parameters() if not only_trainable or p.requires_grad)
