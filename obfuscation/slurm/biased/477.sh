python ./src/run.py --config configs/untarget_bbox_yolo_v3.py --log conf_0.5_dist_0.25_size_0.25_norm_0.05_repeat_2 --cfg-options seed=4217 shuffle=True result_dir=./data/biased/results dataset_dir=./data/biased/datasets log_dir=./data/biased/logs img_dir=./data/biased/images cache_dir=./data/biased/caches criteria_dt.target_max_conf=0.5 criteria_dt.bbox_max_dist=0.25 criteria_dt.perturb_min_size=0.25 max_norm=0.05 attack_samples=100 itr=200
