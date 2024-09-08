import sys
import os
import yaml
import shutil
import extras.logger as logg

from tqdm import tqdm
import cv2

import warnings as wr
wr.filterwarnings("ignore")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import re
import pandas as pd

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data import *
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset

from helper.my_evaluate import *
from helper.detectron_df import *
from helper.my_logger import *

import helper.logger as logg

params = yaml.safe_load(open('params.yaml'))


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


def custom_dataset_function_test():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    split_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}")
    annot_path = os.path.join("data/split",f"v{params['detectron2']['ingest']['dcount']}","labels/val")
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


def pred_best_set():
    output_predict = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    os.makedirs(output_predict, exist_ok = True)

    DatasetCatalog.register("my_dataset_val", custom_dataset_function_test)
    MetadataCatalog.get("my_dataset_val").set(thing_classes = ["person"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['detectron2']['hyps']['config_file']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    # if params['detectron2']['version']['best'] == "v0" or params['detectron2']['weights'] == "pretrained":
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['detectron2']['hyps']['config_file'])
    # else:
    # cfg.MODEL.WEIGHTS = os.path.join("result/detectron2/train/exp{}".format((params['detectron2']['version']['best'])[1]),"weights/model_final.pth")
    cfg.MODEL.WEIGHTS = params['detectron2']['weights']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["detectron2"]["hyps"]["NUM_CLASSES"]
    predictor = DefaultPredictor(cfg)

    test_metadata = MetadataCatalog.get("my_dataset_val")
    dataset_dicts = DatasetCatalog.get("my_dataset_val")
    
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir = output_predict)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

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

    results_logger(det_metrics,my_metrics,output_predict,params)


def copy_to_exp(pred_dir):
    val_path = params['detectron2']['outputs']['val_dir']
    os.makedirs(val_path, exist_ok = True)

    dir = os.listdir(val_path)
    if len(dir) == 0:
        os.makedirs(os.path.join(val_path,"exp1"))
        val_path = os.path.join(val_path,"exp1")
    else:
        latest_path = get_latest_folder_path(val_path)
        num = int(latest_path[-1])
        val_path = latest_path[0:-1] + str(num+1)
        os.makedirs(val_path)

    shutil.copy(os.path.join(pred_dir,"v{}/predict_metrics.json".format(params['detectron2']['ingest']['dcount'])), val_path)
    shutil.copy(os.path.join(pred_dir,"v{}/det2_hyperparamters.json".format(params['detectron2']['ingest']['dcount'])), val_path)