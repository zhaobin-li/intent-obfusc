python ./src/run.py --config configs/mislabel_bbox_faster_rcnn.py --log itr_10_norm_None_repeat_16 --cfg-options seed=24037 shuffle=True result_dir=./data/randomized/results dataset_dir=./data/randomized/datasets log_dir=./data/randomized/logs img_dir=./data/randomized/images cache_dir=./data/randomized/caches itr=10 max_norm=None attack_samples=200
