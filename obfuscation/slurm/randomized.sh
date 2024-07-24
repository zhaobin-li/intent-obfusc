#!/bin/bash

# generate scripts to run all randomized attacks

# start with easiest attack settings
iterations=(200 100 50 10)
max_norms=(None 0.05)
echo iterations: "${iterations[@]}"
echo max_norms: "${max_norms[@]}"

samples=200
repeat=20
echo samples: "${samples}"
echo repeat: "${repeat}"
echo

count=0

for itr in "${iterations[@]}"; do
  for norm in "${max_norms[@]}"; do
    echo

    ./gen_scripts.sh -m faster_rcnn -m yolo_v3 -m retinanet -m ssd_512 -m cascade_rcnn -a vanish_bbox -a mislabel_bbox -a untarget_bbox -d randomized -l itr_"${itr}"_norm_"${norm}" -o itr="${itr}" -o max_norm="${norm}" -o attack_samples="${samples}" -r "${repeat}" -c "${count}"

    ((count += 15 * repeat)) # 5 models and 3 attacks
  done
done

echo
echo count: "$count"
