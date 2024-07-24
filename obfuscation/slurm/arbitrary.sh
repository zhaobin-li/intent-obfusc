#!/bin/bash

# generate scripts to run all deliberate attacks perturbing an arbitrary region

# start with easiest attack settings
bbox_lengths=(0.7 0.5 0.3 0.1)
boundary_distances=(0.01 0.05 0.1 0.2)
echo bbox_lengths: "${bbox_lengths[@]}"
echo boundary_distances: "${boundary_distances[@]}"

max_norms=(None 0.05)
iterations=200
echo max_norms: "${max_norms[@]}"
echo iterations: "${iterations}"

samples=50
repeat=4
echo samples: "${samples}"
echo repeat: "${repeat}"
echo

count=0

for len in "${bbox_lengths[@]}"; do
  for dist in "${boundary_distances[@]}"; do
    for norm in "${max_norms[@]}"; do
      echo

      ./gen_scripts.sh -m faster_rcnn -m yolo_v3 -m retinanet -m ssd_512 -m cascade_rcnn -a vanish_arbitrary -a mislabel_arbitrary -a untarget_arbitrary -d arbitrary -l bbox_"${len}"_dist_"${dist}"_norm_"${norm}" -o criteria_dt.bbox_length="${len}" -o criteria_dt.boundary_distance="${dist}" -o max_norm="${norm}" -o attack_samples="${samples}" -o itr="${iterations}" -r "${repeat}" -c "${count}"

      ((count += 15 * repeat)) # 5 models and 3 attacks
    done
  done
done

echo
echo count: "$count"
