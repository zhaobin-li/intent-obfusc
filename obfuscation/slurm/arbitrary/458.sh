python ./src/run.py --config configs/mislabel_arbitrary_retinanet.py --log bbox_0.7_dist_0.2_norm_0.05_repeat_3 --cfg-options seed=1651 shuffle=True result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images cache_dir=./data/arbitrary/caches criteria_dt.bbox_length=0.7 criteria_dt.boundary_distance=0.2 max_norm=0.05 attack_samples=50 itr=200
