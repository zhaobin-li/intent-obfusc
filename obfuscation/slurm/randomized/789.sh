python ./src/run.py --config configs/mislabel_bbox_ssd_512.py --log itr_100_norm_None_repeat_13 --cfg-options seed=30521 shuffle=True result_dir=./data/randomized/results dataset_dir=./data/randomized/datasets log_dir=./data/randomized/logs img_dir=./data/randomized/images cache_dir=./data/randomized/caches itr=100 max_norm=None attack_samples=200
