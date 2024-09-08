import sys
import os
import yaml
import shutil
import extras.logger as logg

from tqdm import tqdm
import cv2

import warnings as wr
wr.filterwarnings("ignore")

# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

import re
import pandas as pd

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.data import build_detection_test_loader
# from detectron2.data import *
# from detectron2.evaluation import COCOEvaluator
# from detectron2.evaluation import inference_on_dataset


# from helper.my_evaluate import *
# from helper.detectron_df import *
# from extras.my_logger import *

import extras.logger as logg

sys.path.insert(0, 'model/yolov5')
import val

# sys.path.insert(0, 'model/detectron2')
# import pred_det

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/predict.py data/split data/predicted\n'
    )
    sys.exit(1)


def yolov5Model():
    output = os.path.join(sys.argv[2],f"v{params['yolov5']['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    val.run(data='model/yolov5/data/person.yaml', weights=params['yolov5']['weights'], project=params['yolov5']['outputs']['val_dir'])


# def detectron2Model():
#     pred_det.pred_best_set()
#     pred_det.copy_to_exp(sys.argv[2])


def main():
    logger.info('PREDICTING')
    if params['model'] == 'yolov5':
        yolov5Model()
    elif params['model'] == 'detectron2':
        # detectron2Model()
        logger.info('SKIPPING DETECTRON PREDICT STAGE')
    logger.info('PREDICTING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("predict.py")
    params = yaml.safe_load(open('params.yaml'))
    main()