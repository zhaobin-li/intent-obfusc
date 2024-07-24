""" Setup dataset by obtaining and evaluating model predictions and class probability on the ground-truth images """

import logging

import fiftyone as fo
import torch
from fiftyone import ViewField as F
from mmcv import imread
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

from main.utils.bbox import voc_to_fo
from main.utils.detector import (
    get_boxes_labels_scores,
)
from main.utils.detector import get_probas


@torch.no_grad()
def setup(
    compute_map,
    cuda,
    dataset_name,
    gt_samples,
    images_path,
    labels_path,
    log_name,
    min_iou,
    min_score,
    model_checkpoint,
    model_config,
    replace_dataset,
    seed,
    shuffle,
    **kwargs,
):
    logger = logging.getLogger(log_name)

    if dataset_name in fo.list_datasets() and replace_dataset:
        fo.delete_dataset(dataset_name)

    if dataset_name not in fo.list_datasets():
        dataset = fo.Dataset.from_dir(
            name=dataset_name,
            dataset_type=fo.types.COCODetectionDataset,
            label_types=["detections"],  # named as 'ground_truth'
            data_path=images_path,
            labels_path=labels_path,
            max_samples=gt_samples,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        dataset = fo.load_dataset(dataset_name)

    logger.debug(dataset)
    logger.info(dataset.name)

    model = init_detector(model_config, model_checkpoint, device=f"cuda:{cuda}")
    classes = [
        cls for cls in dataset.default_classes if not cls.isnumeric()
    ]  # 80 rather than 91

    logger.debug(model)
    logger.debug(classes)

    logger.info("Starting prediction")

    for sample in tqdm(dataset):
        # uses 'bgr' because Normalize converts to rgb in data pipeline
        img = imread(sample.filepath, "color", "bgr")
        h, w, c = img.shape

        # len(result) = 80, i.e. one array per class containing bboxes n * (xyxy + c)
        bbox_result = inference_detector(model, img)
        boxes, labels, scores = get_boxes_labels_scores(bbox_result)

        probas = get_probas(model)

        detections = []
        for label, score, box, proba in zip(labels, scores, boxes, probas):
            detections.append(
                fo.Detection(
                    label=classes[int(label)],
                    bounding_box=voc_to_fo(box, w, h),
                    confidence=score,
                    label_idx=int(label),
                    proba=proba.tolist(),
                )
            )

        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    # only include predicted bboxes with confidence > min_score
    dataset = dataset.filter_labels("predictions", F("confidence") > min_score)
    dataset.save()

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        iou=min_iou,
        compute_mAP=compute_map,
    )

    return dataset, results
