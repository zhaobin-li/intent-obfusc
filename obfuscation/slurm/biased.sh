#!/bin/bash

# generate scripts to run all deliberate attacks perturbing selected bboxes meeting various criteria

# start with easiest attack settings
max_confs=(None 0.5)
max_dists=(None 0.25)
min_sizes=(None 0.25)
echo max_confs: "${max_confs[@]}"
echo max_dists: "${max_dists[@]}"
echo min_sizes: "${min_sizes[@]}"

max_norms=(None 0.05)
iterations=200
echo max_norms: "${max_norms[@]}"
echo iterations: "${iterations}"

samples=100
repeat=2
echo samples: "${samples}"
echo repeat: "${repeat}"
echo

count=0

for norm in "${max_norms[@]}"; do
  for conf in "${max_confs[@]}"; do
    for dist in "${max_dists[@]}"; do
      for size in "${min_sizes[@]}"; do
        echo

        ./gen_scripts.sh -m faster_rcnn -m yolo_v3 -m retinanet -m ssd_512 -m cascade_rcnn -a vanish_bbox -a mislabel_bbox -a untarget_bbox -d biased -l conf_"${conf}"_dist_"${dist}"_size_"${size}"_norm_"${norm}" -o criteria_dt.target_max_conf="${conf}" -o criteria_dt.bbox_max_dist="${dist}" -o criteria_dt.perturb_min_size="${size}" -o max_norm="${norm}" -o attack_samples="${samples}" -o itr="${iterations}" -r "${repeat}" -c "${count}"

        ((count += 15 * repeat)) # 5 models and 3 attacks
      done
    done
  done
done

echo
echo count: "$count"
