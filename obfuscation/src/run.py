""" Intent obfuscating attack

In order, run
    1. setup.py
    2. attack.py
    3. export.py
and export attacked dataset
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import fiftyone as fo
from mmcv import Config, DictAction

from main.attack import attack
from main.export import export
from main.setup import setup
from main.utils.misc import get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent obfuscating attack")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/vanish_retinanet_test.py",
        help="attack config in ./configs directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=datetime.now().strftime("%y%m%d_%H%M%S"),
        help="log name",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="modify config according to https://mmdetection.readthedocs.io/en/v2.25.0/tutorials/config.html",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.log_name = f"{args.log}_{Path(args.config).stem}"
    logger = get_logger(cfg.log_dir, cfg.log_name, cfg.log_level)

    cfg.dataset_name = f"{cfg.log_name}_{cfg.dataset_name}"

    logger.info(cfg)
    logger.debug(fo.config)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_name = os.path.join(cfg.cache_dir, Path(cfg.model_checkpoint).stem)

    try:
        # save and reuse evaluation results per model
        if not os.path.isdir(cache_name):  # cache_name is directory
            dataset = setup(**cfg)[0]
            if not os.path.isdir(cache_name):  # check again to avoid clashes
                dataset.export(
                    export_dir=cache_name,
                    dataset_type=fo.types.FiftyOneDataset,
                    export_media=False,
                    rel_dir=cfg.images_path,
                )
                logger.info(f"Exported {cache_name=}!")
        else:
            dataset = fo.Dataset.from_dir(
                dataset_dir=cache_name,
                dataset_type=fo.types.FiftyOneDataset,
                rel_dir=cfg.images_path,
                name=cfg.dataset_name,
            )
            logger.info(f"Imported {cache_name=}!")

        dataset = attack(**cfg)[0]

        # save config
        os.makedirs(cfg.dataset_dir, exist_ok=True)
        save_name = os.path.join(cfg.dataset_dir, f"{cfg.dataset_name}")

        cfg.dump(f"{save_name}.py")
        logger.info(f"Dumping {save_name=}...")

        # export attacked slice including evaluations
        # https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#fiftyonedataset
        attack_slice = dataset.match_tags("attack")

        attack_slice.export(
            export_dir=save_name,
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False,
            rel_dir=cfg.images_path,
        )

        export(**cfg)

        if cfg.launch_app:
            # ssh -N -L 5151:127.0.0.1:5151 [<username>@]<hostname>
            session = fo.launch_app(dataset, remote=True)
            session.wait()

    except Exception as e:
        logger.exception(e)
