def get_arbitrary_bbox(
    ground_truth,
    criteria_dt,
    data,
    rng,
):
    """Create arbitrary bbox or return None

    Inputs: bbox_length and boundary_distance in criteria_dt are normalized image units
    Returns: [start_x, start_y, end_x, end_y] are scaled image units
    """

    # normalized image units ------------------------------
    arbitrary_bbox_rad = criteria_dt["bbox_length"] / 2
    boundary_distance = criteria_dt["boundary_distance"]

    # resized image units ------------------------------
    target_idx = rng.choice(ground_truth["tp_idxs"])
    target_bbox = data["gt_bboxes"][0][target_idx]

    start_x, start_y, end_x, end_y = target_bbox.tolist()
    target_center_x, target_center_y = (end_x + start_x) / 2, (end_y + start_y) / 2
    target_rad_x, target_rad_y = (end_x - start_x) / 2, (end_y - start_y) / 2

    # img_shape is (h, w, c): mmdetection/mmdet/datasets/pipelines/formatting.py:289
    # Image is padded bottom and right (mmcv/image/geometric.py:495).
    # We don't use pad_shape since we would like arbitrary bbox to be within image excluding paddind
    img_y, img_x, *_ = data["img_metas"][0]["img_shape"]

    # try to construct an arbitrary bbox in every perturb_direction
    # and returns the 1st arbitrary bbox within image bounds,
    perturb_directions = ["left", "right", "top", "bottom"]
    rng.shuffle(perturb_directions)

    for perturb_dir in perturb_directions:
        if perturb_dir == "left" or perturb_dir == "right":
            # distance in width between arbitrary and target bbox centers in resized image units
            distance_x = target_rad_x + (boundary_distance + arbitrary_bbox_rad) * img_x
            if perturb_dir == "left":
                center_x = target_center_x - distance_x
            else:
                center_x = target_center_x + distance_x

            start_x = center_x - arbitrary_bbox_rad * img_x
            end_x = center_x + arbitrary_bbox_rad * img_x

            # arbitrary bbox center is aligned to target center in height
            start_y = target_center_y - arbitrary_bbox_rad * img_y
            end_y = target_center_y + arbitrary_bbox_rad * img_y

        else:  # similar to "left" and "right" besides switching width and height
            distance_y = target_rad_y + (boundary_distance + arbitrary_bbox_rad) * img_y
            if perturb_dir == "top":
                # y increases downwards
                center_y = target_center_y - distance_y
            else:
                center_y = target_center_y + distance_y

            start_y = center_y - arbitrary_bbox_rad * img_y
            end_y = center_y + arbitrary_bbox_rad * img_y

            start_x = target_center_x - arbitrary_bbox_rad * img_x
            end_x = target_center_x + arbitrary_bbox_rad * img_x

        within_img = (0 <= start_x <= end_x <= img_x) and (
            0 <= start_y <= end_y <= img_y
        )
        if within_img:
            return perturb_dir, target_idx, [start_x, start_y, end_x, end_y]

    return None
