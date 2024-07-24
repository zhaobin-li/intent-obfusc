# dataset
gt_samples = 5000

attack_samples = None
shuffle = False

seed = None
replace_dataset = True

compute_map = False

# device
cuda = 0

# app
launch_app = False

# attack
itr = 10
max_norm = 0.1

min_iou = 0.3
min_score = 0.3

non_overlap = True
attack_bbox = "predictions"  # or "ground_truth"

adversarial_target = "vanish"
perturb_fun = "perturb_inside"  # only option now

arbitrary_bbox = False
criteria_dt = dict(
    target_max_conf=None, perturb_min_size=None, bbox_max_dist=None
)  # main/utils/criteria.py

# data
result_dir = "./data/test/results"
dataset_dir = "./data/test/datasets"

cache_dir = "./data/test/caches"
log_dir = "./data/test/logs"

img_dir = "./data/test/images"
viz_pgd = False

# misc
log_level = "DEBUG"
