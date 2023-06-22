def adjust_learning_rate(optimizer, base_lr, min_lr, step, total_steps, warm_up_steps, lr_decay):

    if step < warm_up_steps:
        now_lr = min_lr + (base_lr - min_lr) * step / warm_up_steps
    else:
        step = step - warm_up_steps
        total_steps = total_steps - warm_up_steps
        now_lr = min_lr + (base_lr - min_lr) * (1 - step / (total_steps + 1))**lr_decay

    for param_group in optimizer.param_groups:
        if "encoder" in param_group["name"]:
            param_group['lr'] = (now_lr - min_lr) * 0.1 + min_lr
            # param_group['lr'] = now_lr * 0.1
        else:
            param_group['lr'] = now_lr
    return now_lr


def setup_optimizer_param_groups(model, weight_decay, base_lr):
    params = []
    memo = set()
    no_wd_keys = ['absolute_pos_embed', 'relative_position_bias_table', 'relative_emb_v', 'conv_out']

    for key, value in model.named_parameters():
        if value in memo:
            continue
        if not value.requires_grad:
            continue
        memo.add(value)
        wd = weight_decay
        if len(value.shape) == 1:
            if 'bias' in key or 'encoder.' not in key:
                wd = 0.
        else:
            for no_wd_key in no_wd_keys:
                if no_wd_key in key:
                    wd = 0.
                    break
        params += [{
            "params": [value],
            "lr": base_lr,
            "weight_decay": wd,
            "name": key
        }]

    return params
