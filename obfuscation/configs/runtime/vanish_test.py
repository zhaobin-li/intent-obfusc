# _base_ = ["./vanish_arbitrary.py"]
# criteria_dt = dict(bbox_length=0.5, boundary_distance=0.01)

_base_ = ["./vanish_bbox.py"]
# criteria_dt = dict(
#     target_max_conf=0.5, perturb_min_size=0.25, bbox_max_dist=0.25
# )  # main/utils/criteria.py

max_norm = None

gt_samples = 100
attack_samples = 10

shuffle = True
seed = 5151

launch_app = True
viz_pgd = True
