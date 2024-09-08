import os
import sys
import re
import pandas as pd
import yaml, shutil
import json
import extras.logger as logg
import shutil

import time

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/compare.py data/trained data/compared data/predicted\n'
    )
    sys.exit(1)


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

def get_metrics(dir):
    # path = [sorted(Path(dir).iterdir(), key=os.path.getmtime,reverse=True)][0][0]
    path = get_latest_folder_path(dir)
    if int(path[-1]) == 0:
        path = path[:-1]
    # print("Path i got : ", path)
    metrics_df = []
    for ex in os.listdir(path):
        if ex.endswith('.csv') and 'metrics' in ex:
            metrics_path = os.path.join(path,ex)
            metrics_df = pd.read_csv(metrics_path)
        else:
            pass
    return metrics_df

def get_best_model(met_dir,flag):
    # dst=''
    if flag == False:
        path = get_latest_folder_path(met_dir)
        if int(path[-1]) == 0:
            path = path[:-1]
        
        path = str(path) + '/weights/best.pt'

    else:
        if len(os.listdir(met_dir)) > 1:
            path = get_latest_folder_path(met_dir)
            path = str(path) + '/weights/best.pt'
        else:
            path = params['yolov5']['weights']
    return path

def compare_metrics(val_met,pred_met,met_dir):
    best_model_path = ''
    val_F1 = val_met['F1-Score'][0]
    pred_F1 = pred_met['F1-Score'][0]
    # val_mAP95 = val_met['mAP50-95'][0]
    # pred_mAP95 = pred_met['mAP50-95'][0]
    #False - Validated model is better
    #True - predicted model is better
    flag = True if val_F1 > pred_F1 else False
    best_model_path = get_best_model(met_dir,flag)
    return flag, best_model_path


def compare(metrics_best,metrics_new,predict_path,train_path):
    best_f1 = metrics_best["F1"]
    best_ap = metrics_best["AP"]
    new_f1 = metrics_new["F1"]
    new_ap = metrics_new["AP"]
    print("best_f1",best_f1)
    print("new_f1",new_f1)
    print("best_ap",best_ap)
    print("new_ap",new_ap)

    flag = False

    if best_f1 < new_f1:
        best_model = os.path.join(sys.argv[1],"v{}/model_final.pth".format(params['detectron2']['ingest']['dcount']))
        dst = params['detectron2']['weights']
        shutil.copyfile(best_model, dst)
        best_model_path = dst
        flag = True
        payload = {"flag":flag,"best_model_path":best_model_path}
        with open('register.json', 'w') as fout:
            fout.write(json.dumps(payload, indent = len(payload)))

        f = open ("register.json", "r")
        print(json.loads(f.read()))

    dcount = params['detectron2']['ingest']['dcount'] + 1
    params['detectron2']['ingest']['dcount'] = dcount
    with open('params.yaml', 'w') as file:
        outputs = yaml.dump(params, file, sort_keys=False)
    best_model = params['detectron2']['weights']

    with open('new_metrics.json', 'w') as fout:
        fout.write(json.dumps(metrics_new, indent = len(metrics_new)))
    time.sleep(5)

    f = open ("new_metrics.json", "r")
    print('\n\n\n')
    print(json.loads(f.read()))
    print('\n\n\n')

    with open('prev_metrics.json', 'w') as fout:
        fout.write(json.dumps(metrics_best, indent = len(metrics_best)))
    time.sleep(5)

    f = open ("prev_metrics.json", "r")
    print('\n\n\n')
    print(json.loads(f.read()))
    print('\n\n\n')

def get_new_model_metrices(val_met):
    precision = val_met['Precision'][0]
    recall = val_met['Recall'][0]
    f1_score = val_met['F1-Score'][0]
    mAP50 = val_met['mAP50'][0]
    mAP50_95 = val_met['mAP50-95'][0]
    return precision, recall, f1_score, mAP50, mAP50_95


def make_dict(precision, recall, f1_score, mAP50, mAP50_95):
    return {"Precision":precision,"Recall":recall,"F1-score":f1_score,"mAP50":mAP50,"mAP50_95":mAP50_95}


def yolov5Model():

    print("Reading Train dir from : ",params['yolov5']['outputs']['train_dir'])
    train_dir = params['yolov5']['outputs']['train_dir']
    val_dir = params['yolov5']['outputs']['val_dir']
    val_met = get_metrics(train_dir)
    pred_met = get_metrics(val_dir)
    print("New Model")
    print(val_met)
    print("Previous Best Model")
    print(pred_met)
    
    flag, best_model = compare_metrics(val_met,pred_met,train_dir)
    precision_val, recall_val, f1_score_val, mAP50_val, mAP50_95_val = get_new_model_metrices(val_met)
    precision_pred, recall_pred, f1_score_pred, mAP50_pred, mAP50_95_pred = get_new_model_metrices(pred_met)

    val_dict = make_dict(precision_val, recall_val, f1_score_val, mAP50_val, mAP50_95_val)
    pred_dict = make_dict(precision_pred, recall_pred, f1_score_pred, mAP50_pred, mAP50_95_pred)
    
    print(val_dict)
    print(pred_dict)
    with open('new_metrics.json', 'w') as fout:
        fout.write(json.dumps(val_dict, indent = len(val_dict)))
    
    time.sleep(5)

    f = open ("new_metrics.json", "r")
    print('\n\n\n')
    print(json.loads(f.read()))
    print('\n\n\n')
    
    with open('prev_metrics.json', 'w') as fout:
        fout.write(json.dumps(pred_dict, indent = len(pred_dict)))

    time.sleep(5)

    f = open ("prev_metrics.json", "r")
    print('\n\n\n')
    print(json.loads(f.read()))
    print('\n\n\n')

    if flag:
        dst = params['yolov5']['weights']
        shutil.copyfile(best_model, dst)
        print("Copied weights ", best_model, dst)  # copy src to dst

        payload = {"flag":flag,"best_model_path":best_model}
        with open('register.json', 'w') as fout:
            fout.write(json.dumps(payload, indent = len(payload)))
    params['yolov5']['weights'] = best_model
    yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)


def detectron2Model():
    predict_path = os.path.join(sys.argv[3],f"v{params['detectron2']['ingest']['dcount']}")
    train_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}")
    datastore = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    os.makedirs(datastore, exist_ok = True)

    metrics_best_path = os.path.join(predict_path,"predict_metrics.json")
    metrics_new_path = os.path.join(train_path,"predict_metrics.json")

    result_train_path = params['detectron2']['outputs']['train_dir']
    result_val_path = params['detectron2']['outputs']['val_dir']
    f1 = open (metrics_best_path, "r")
    metrics_best = json.loads(f1.read())

    f2 = open (metrics_new_path, "r")
    metrics_new = json.loads(f2.read())

    #Compare metrics
    compare(metrics_best, metrics_new,result_val_path,result_train_path)

    shutil.rmtree('.dvc/cache', ignore_errors=True) 


def main():
    logger.info('COMPARING')
    if params['model'] == 'yolov5':
        yolov5Model()
    elif params['model'] == 'detectron2':
        detectron2Model()
    logger.info('COMPARING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("compare.py")
    params = yaml.safe_load(open('params.yaml'))
    output = os.path.join(sys.argv[2],f"v{params[params['model']]['ingest']['dcount']}",'images')
    os.makedirs(output, exist_ok=True)
    main()

    # 'yolo_v5_model_ede1'
    # 'detectron2_model_ede1'