python ./src/run.py --config configs/mislabel_bbox_faster_rcnn.py --log itr_100_norm_0.05_repeat_16 --cfg-options seed=16249 shuffle=True result_dir=./data/randomized/results dataset_dir=./data/randomized/datasets log_dir=./data/randomized/logs img_dir=./data/randomized/images cache_dir=./data/randomized/caches itr=100 max_norm=0.05 attack_samples=200
