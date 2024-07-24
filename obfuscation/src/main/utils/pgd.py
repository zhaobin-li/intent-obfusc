import copy

import torch
from mmdet.models import TwoStageDetector, SingleStageDetector


def pgd(
    model_,
    data_,
    backward_loss,
    perturb_bbox,
    itr,
    lr,
    lower_bound,
    upper_bound,
    perturb_fun,
    max_norm,
):
    perturb_fun = globals()[perturb_fun]  # calling locals() should be the same

    if max_norm is not None:  # max_norm units proportion
        max_norm = max_norm * (upper_bound - lower_bound)

        lower_bound = torch.maximum(data_["img"] - max_norm, lower_bound)
        upper_bound = torch.minimum(data_["img"] + max_norm, upper_bound)

    data = copy.deepcopy(data_)
    for i in range(itr):
        model = copy.deepcopy(model_)  # avoid changing the model during training
        data["img"] = data["img"].requires_grad_()

        # losses dict[str, Tensor]
        if isinstance(model, TwoStageDetector):
            # do not pass `rescale=True` in training
            # since we match predictions with rescaled gt_bboxes
            losses = model(return_loss=True, **data)
        elif isinstance(model, SingleStageDetector):
            losses = model(return_loss=True, **data)
        else:
            raise RuntimeError

        model.zero_grad()
        backward_loss(losses).backward()

        delta = perturb_fun(data, perturb_bbox)
        data["img"] = data["img"] - lr * delta

        data["img"] = torch.clamp(data["img"], min=lower_bound, max=upper_bound)
        data["img"] = data["img"].detach_()
    return data


def perturb_inside(data, perturb_bbox):
    # image (..., H, W)
    start_x, start_y, end_x, end_y = list(map(round, perturb_bbox))

    delta = torch.zeros_like(data["img"])
    delta[..., start_y:end_y, start_x:end_x] = data["img"].grad.sign()[
        ..., start_y:end_y, start_x:end_x
    ]

    return delta
