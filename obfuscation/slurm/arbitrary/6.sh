python ./src/run.py --config configs/mislabel_arbitrary_faster_rcnn.py --log bbox_0.7_dist_0.01_norm_None_repeat_1 --cfg-options seed=701 shuffle=True result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images cache_dir=./data/arbitrary/caches criteria_dt.bbox_length=0.7 criteria_dt.boundary_distance=0.01 max_norm=None attack_samples=50 itr=200
