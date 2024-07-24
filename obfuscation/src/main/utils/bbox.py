# Ref: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
# voc: [x_min, y_min, x_max, y_max]
# yolo: [x_center, y_center, width, height] normalized
# fo: [x_min, y_min, width, height] normalized
from itertools import permutations

from main.utils.arbitrary import get_arbitrary_bbox
from main.utils.criteria import satisfy_criteria


def voc_to_fo(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


def fo_to_voc(box, w, h):
    x1, y1, w1, h1 = box
    return [x1 * w, y1 * h, (w1 + x1) * w, (h1 + y1) * h]


def fo_to_yolo(box):
    x1, y1, w1, h1 = box
    return [x1 + w1 / 2, y1 + h1 / 2, w1, h1]


def sample_bbox_pair(ground_truth, non_overlap, arbitrary_bbox, criteria_dt, data, rng):
    """sample perturb and target pairs according to criteria_dt or return None"""
    if arbitrary_bbox:
        return get_arbitrary_bbox(
            ground_truth,
            criteria_dt,
            data,
            rng,
        )

    pairs = list(permutations(ground_truth["tp_idxs"], 2))  # no replacement
    rng.shuffle(pairs)  # in_place

    for perturb_idx, target_idx in pairs:
        if satisfy_criteria(
            ground_truth, perturb_idx, target_idx, non_overlap, criteria_dt
        ):
            return perturb_idx, target_idx, data["gt_bboxes"][0][perturb_idx].tolist()
    return None
