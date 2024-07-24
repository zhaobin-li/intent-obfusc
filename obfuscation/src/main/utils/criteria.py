from math import sqrt


def intersect(self, other):
    # self and other: x1, y1, x2, y2 based on https://stackoverflow.com/questions/40795709/checking-whether-two
    # -rectangles-overlap-in-python-using-two-bottom-left-corners return not (self.top_right.x < other.bottom_left.x
    # or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y >
    # other.top_right.y)

    return not (
        self[2] < other[0]
        or self[0] > other[2]
        or self[1] > other[3]
        or self[3] < other[1]
    )  # change sign as y increases downwards


def min_distance(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
    """based on # https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
    get shortest distance between two bbox edges"""
    # larger rectangle enveloping both
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)
    x2 = max(x2_1, x2_2)
    y2 = max(y2_1, y2_2)

    inner_width = max(0, (x2 - x1) - (x2_1 - x1_1) - (x2_2 - x1_2))
    inner_height = max(0, (y2 - y1) - (y2_1 - y1_1) - (y2_2 - y1_2))

    return sqrt(inner_width**2 + inner_height**2)


def target_max_conf(ground_truth, perturb_idx, target_idx, max_conf):
    return ground_truth["conf"][target_idx] < max_conf


def perturb_min_size(ground_truth, perturb_idx, target_idx, min_size):
    return ground_truth["rel_size"][perturb_idx] > min_size


def bbox_max_dist(ground_truth, perturb_idx, target_idx, max_dist):
    return (
        min_distance(
            *ground_truth["rel_box"][perturb_idx], *ground_truth["rel_box"][target_idx]
        )
        < max_dist
    )


def satisfy_criteria(ground_truth, perturb_idx, target_idx, non_overlap, criteria_dt):
    """check whether perturb and target bbox pair satisfy non_overlap and all selection criteria included in
    criteria_dt"""
    if non_overlap and intersect(
        ground_truth["boxes"][perturb_idx], ground_truth["boxes"][target_idx]
    ):
        return False

    for (
        cri,
        cri_arg,
    ) in criteria_dt.items():  # criteria_dt contains function names as string
        cri_fun = globals()[cri]
        if cri_arg is not None and not cri_fun(
            ground_truth, perturb_idx, target_idx, cri_arg
        ):
            return False
    return True
