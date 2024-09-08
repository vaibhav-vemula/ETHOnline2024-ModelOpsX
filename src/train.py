import sys
import os
import yaml
import extras.logger as logg
import shutil
from tqdm import tqdm
import cv2
import re
import pandas as pd

import warnings as wr
wr.filterwarnings("ignore")

# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
# from detectron2.data.catalog import DatasetCatalog
# from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.engine import DefaultTrainer
# from detectron2.evaluation import COCOEvaluator
# from detectron2.utils.visualizer import Visualizer

# from helper.my_evaluate import *
# from helper.my_logger import *
# from helper.detectron_df import *
import extras.logger as logg

sys.path.insert(0, 'model/yolov5')
import train
# sys.path.insert(0, 'model/detectron2')
# import train_det


if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/split data/trained data/augmented\n'
    )
    sys.exit(1)


def yolov5Model():
    output = os.path.join(sys.argv[2],f"v{params['yolov5']['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    args = params['yolov5']['hyps']
    wt = params['yolov5']['weights']
    train.run(data='person.yaml', imgsz=320, weights=wt, project=params['yolov5']['outputs']['train_dir'], **args)


# def detectron2Model():
#     cfg = train_det.train_set()
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
#     trainer = DefaultTrainer(cfg) 
#     trainer.resume_or_load(resume = False)
#     trainer.train()
#     train_det.predict_set(cfg)
#     train_det.copy_to_exp()


def main():
    logger.info('TRAINING')
    if params['model'] == 'yolov5':
        yolov5Model()
    elif params['model'] == 'detectron2':
        # detectron2Model()
        logger.info('SKIPPING DETECTRON TRAIN STAGE')
    logger.info('TRAINING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("train.py")
    params = yaml.safe_load(open('params.yaml'))
    main()