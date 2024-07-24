_base_ = ["./vanish_bbox.py"]

# bbox_length int: arbitrary bbox width and height in original image pixel units
# boundary_distance int: distance between arbitrary bbox and target bbox in original image pixel units
arbitrary_bbox = True
criteria_dt = dict(bbox_length=0.5, boundary_distance=0.1)
