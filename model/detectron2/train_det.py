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

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer

from helper.my_evaluate import *
from helper.my_logger import *
from helper.detectron_df import *
import helper.logger as logg

params = yaml.safe_load(open('params.yaml'))
def custom_dataset_function_train():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    split_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}")
    # annot_path = os.path.join("data/split",f"v{params['detectron2']['ingest']['dcount']}","labels/train")
    img_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}","images/train")

    dataframe = pd.read_pickle(os.path.join(split_path,"v{}_train.pkl".format(params['detectron2']['ingest']['dcount'])))
    dataframe["name"] = [x.replace("labels","images") for x in dataframe["name"]]

    # old_fname = os.path.join(dataframe["name"][0]) + ".jpg"
    old_fname = os.path.join(img_path,dataframe["name"][0]) + ".jpg"
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        fname = os.path.join(img_path, row["name"]) + ".jpg"
        # fname = os.path.join(row["name"]) + ".jpg"
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]

        if old_fname != fname:
            img = cv2.imread(old_fname)
            dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})
            annotations = []

        annotations.append(
            {"bbox":[xmin,ymin,xmax,ymax],
            'bbox_mode': 0, 
            'area': (xmax - xmin) * (ymax - ymin), 
            'segmentation': [],
            'category_id':0})
        old_fname = fname

    img = cv2.imread(old_fname)
    dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})
    return dataset


def custom_dataset_function_test():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    split_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}")
    # annot_path = os.path.join("data/split",f"v{params['detectron2']['ingest']['dcount']}","labels/val")
    img_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}","images/val")
    
    # dataframe = creatingInfoData(annot_path)
    dataframe = pd.read_pickle(os.path.join(split_path,"v{}_val.pkl".format(params['detectron2']['ingest']['dcount'])))
    dataframe["name"] = [x.replace("labels","images") for x in dataframe["name"]]

    old_fname = os.path.join(img_path,dataframe["name"][0]) + ".jpg"
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        fname = os.path.join(img_path, row["name"]) + ".jpg"
        # fname = os.path.join(row["name"]) + ".jpg"
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]
        if old_fname != fname:
            img = cv2.imread(old_fname)
            dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})
            annotations = []
        annotations.append(
            {"bbox":[xmin,ymin,xmax,ymax],
            'bbox_mode': 0, 
            'area': (xmax - xmin) * (ymax - ymin), 
            'segmentation': [],
            'category_id':0})
        old_fname = fname

    img = cv2.imread(old_fname)
    dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})
    return dataset


def get_latest_folder_path(dir):
    num_li = []
    regex = re.compile(r'\d+')
    # print("Received dir : ", dir)
    folderlist = os.listdir(dir)
    # print("folderlist :", folderlist)
    for filename in folderlist:
        find_res = regex.findall(filename)
        # print("Found nums :", find_res)
        num = find_res[0] if len(find_res) == 1 else 0
        num_li.append(int(num))
    latest_folder = "exp" + str(sorted(num_li, reverse=True)[0])
    # print(path + "/" + latest_folder)
    path = os.path.join(dir, latest_folder)
    return path


def train_set():
    train_path = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    aug_path = os.path.join(sys.argv[3],f"v{params['detectron2']['ingest']['dcount']}")
    os.makedirs(train_path, exist_ok = True)
    DatasetCatalog.register("my_dataset_train", custom_dataset_function_train)
    MetadataCatalog.get("my_dataset_train").set(thing_classes = ["person"])
    DatasetCatalog.register("my_dataset_val", custom_dataset_function_test)
    MetadataCatalog.get("my_dataset_val").set(thing_classes = ["person"])
    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["detectron2"]["hyps"]["config_file"]))
    # if params['detectron2']['version']['best'] == "v0" or params['detectron2']['weights'] == "pretrained":
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params["detectron2"]["hyps"]["config_file"])
    # else:
    cfg.MODEL.WEIGHTS = params['detectron2']['weights']
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = params["detectron2"]["hyps"]["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = params["detectron2"]["hyps"]["IMS_PER_BATCH"]
    cfg.SOLVER.BASE_LR = params["detectron2"]["hyps"]["BASE_LR"]
    cfg.SOLVER.WARMUP_ITERS = params["detectron2"]["hyps"]["WARM_UP_ITERS"]
    cfg.SOLVER.MAX_ITER = params["detectron2"]["hyps"]["MAX_ITER"] #adjust up if val mAP is still rising, adjust down if overfit
    # cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = params["detectron2"]["hyps"]["GAMMA"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params["detectron2"]["hyps"]["BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["detectron2"]["hyps"]["NUM_CLASSES"]
    cfg.TEST.EVAL_PERIOD = params["detectron2"]["hyps"]["EVAL_PERIOD"]
    cfg.OUTPUT_DIR = train_path
    return cfg


def predict_set(cfg):
    train_path = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    cfg.OUTPUT_DIR = train_path
    cfg.MODEL.WEIGHTS = os.path.join(train_path,"model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["detectron2"]["hyps"]["SCORE_THRESH_TEST"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["detectron2"]["hyps"]["NUM_CLASSES"]
    
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir = train_path)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    test_metadata = MetadataCatalog.get("my_dataset_val")
    dataset_dicts = DatasetCatalog.get("my_dataset_val")

    d1 = next(iter(eval_results.items()))
    d2 = next(iter(eval_results.values()))
    det_metrics = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs'],'APm':d2['APm'],'APl':d2['APl']}
    annot_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}","labels/val")
    gt_df = creatingInfoData(annot_path)
    gt_df["name"] = [x["name"].split("/")[-1] for index,x in gt_df.iterrows()]

    metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}

    img_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}","images/val")
    for filename in tqdm(os.listdir(annot_path)):
        img = os.path.join(img_path, filename).replace("xml","jpg")
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"].pred_boxes.to('cpu'))
        gt_df_sub = gt_df[gt_df["name"] == filename.split(".")[0]]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)
    
    my_metrics = evaluate(metrics)
    results_logger(det_metrics,my_metrics,train_path,params)
    

def copy_to_exp(train_dir):
    train_path = params['detectron2']['outputs']['train_dir']
    os.makedirs(train_path, exist_ok = True)

    dir = os.listdir(train_path)
    if len(dir) == 0:
        os.makedirs(os.path.join(train_path,"exp1"))
        train_path = os.path.join(train_path,"exp1")
    else:
        latest_path = get_latest_folder_path(train_path)
        num = int(latest_path[-1])
        train_path = latest_path[0:-1] + str(num+1)
        os.makedirs(train_path)

    train_weights_path = os.path.join(train_path,"weights")
    os.makedirs(train_weights_path, exist_ok = True)
    if os.path.exists(os.path.join(train_dir,"v{}/model_final.pth".format(params['detectron2']['ingest']['dcount']))):
        shutil.copy(os.path.join(train_dir,"v{}/model_final.pth".format(params['detectron2']['ingest']['dcount'])), train_weights_path)
    shutil.copy(os.path.join(train_dir,"v{}/metrics.json".format(params['detectron2']['ingest']['dcount'])), train_path)
    shutil.copy(os.path.join(train_dir,"v{}/predict_metrics.json".format(params['detectron2']['ingest']['dcount'])), train_path)
    shutil.copy(os.path.join(train_dir,"v{}/det2_hyperparamters.json".format(params['detectron2']['ingest']['dcount'])), train_path)
