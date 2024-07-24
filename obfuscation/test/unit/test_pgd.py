import numpy as np
import torch
from mmcv import imread

from main.utils.detector import get_train_data, init_train_detector
from main.utils.loss import get_yolo_v3_vanish_loss
from main.utils.pgd import pgd

device = torch.device("cpu")
model = init_train_detector(
    "../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py",
    "../mmdetection/checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    device=device,
)

# yolo don't normalize
lower_bound = torch.tensor(0).repeat(3, 1, 1).to(device)
upper_bound = torch.tensor(1).repeat(3, 1, 1).to(device)

img = imread("../mmdetection/demo/demo.jpg", "color", "bgr")

data = get_train_data(img, np.float32([[0, 1, 3, 4]]), np.int_([0]), model.cfg, device)
original_img_tensor = data["img"].clone()


def test_pgd_perturb_inside_zero_bbox():
    """check pgd with no perturbation does not change image"""
    pgd_data = pgd(
        model_=model,
        data_=data,
        backward_loss=get_yolo_v3_vanish_loss,
        perturb_bbox=[0, 0, 0, 0],
        itr=1,
        lr=0.5,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        perturb_fun="perturb_inside",
        max_norm=None,
    )

    assert torch.equal(original_img_tensor, pgd_data["img"])


def test_pgd_perturb_inside_max_norm():
    """check pgd max-norm constraint"""

    pgd_data = pgd(
        model_=model,
        data_=data,
        backward_loss=get_yolo_v3_vanish_loss,
        perturb_bbox=[3, 5, 10, 13],
        itr=1,
        lr=0.5,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        perturb_fun="perturb_inside",
        max_norm=0.123,  # max_norm units proportion
    )

    assert torch.all(torch.ge(pgd_data["img"], lower_bound)) and torch.all(
        torch.le(pgd_data["img"], upper_bound)
    )

    assert torch.all(
        torch.le(
            torch.abs(original_img_tensor - pgd_data["img"]),
            max_norm * (upper_bound - lower_bound) + 1e-08,
        )
    )


def test_pgd_perturb_no_max_norm():
    """check pgd within [0,1]"""
    pgd_data = pgd(
        model_=model,
        data_=data,
        backward_loss=get_yolo_v3_vanish_loss,
        perturb_bbox=[3, 5, 10, 13],
        itr=1,
        lr=2,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        perturb_fun="perturb_inside",
        max_norm=None,
    )

    assert torch.all(torch.ge(pgd_data["img"], lower_bound)) and torch.all(
        torch.le(pgd_data["img"], upper_bound)
    )
