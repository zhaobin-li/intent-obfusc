python ./src/run.py --config configs/mislabel_bbox_retinanet.py --log conf_None_dist_None_size_0.25_norm_0.05_repeat_2 --cfg-options seed=14041 shuffle=True result_dir=./data/biased/results dataset_dir=./data/biased/datasets log_dir=./data/biased/logs img_dir=./data/biased/images cache_dir=./data/biased/caches criteria_dt.target_max_conf=None criteria_dt.bbox_max_dist=None criteria_dt.perturb_min_size=0.25 max_norm=0.05 attack_samples=100 itr=200
