python ./src/run.py --config configs/mislabel_arbitrary_cascade_rcnn.py --log bbox_0.3_dist_0.01_norm_None_repeat_3 --cfg-options seed=25664 shuffle=True result_dir=./data/arbitrary/results dataset_dir=./data/arbitrary/datasets log_dir=./data/arbitrary/logs img_dir=./data/arbitrary/images cache_dir=./data/arbitrary/caches criteria_dt.bbox_length=0.3 criteria_dt.boundary_distance=0.01 max_norm=None attack_samples=50 itr=200
