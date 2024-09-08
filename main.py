import streamlit as st
import yaml
import os
import model.yolov5.detect as detect
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import json
import cv2

# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

params = yaml.safe_load(open('params.yaml'))

def pipeline():
    st.subheader('Choose Dataset')
    opts = os.listdir('buffer')
    opts.sort()
    option = st.selectbox('',opts)

    if st.button('Run Pipeline'):
        st.subheader('Running YoloV5 Pipeline..........')
        shutil.rmtree('.dvc/cache', ignore_errors=True) 
        
        params["yolov5"]['ingest']['dcount'] = params["yolov5"]['ingest']['dcount'] +1
        params["yolov5"]['ingest']['dpath'] = option
        yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)
        
        if not os.system("dvc repro"):
            st.success('Pipeline executed successfully')
            st.balloons()
            st.snow()
            show_metrics()
        else:
            st.error("Pipleine execution failed")

def show_metrics():
    params = yaml.safe_load(open('params.yaml'))
    dcou = params["yolov5"]['ingest']['dcount']
    if dcou != 0:
        if params["yolov5"]["weights"] == "pretrained/best.pt":
            bbb = 1
        else:
            if params["yolov5"]["weights"].split("/")[3] == 'exp':
                bbb=1
            else:
                bbb = params["yolov5"]["weights"].split("/")[3]
                bbb = bbb[-1]

        if int(bbb) == int(dcou):
            if bbb == 1:
                # st.write('current model is  runs/val/exp')
                # st.write('prev model is  runs/train/exp')
                prev_best_model = 'runs/yolov5/train/exp'
                current_model = 'runs/yolov5/val/exp'
            else:
                # st.write('current model is  runs/val/exp'+str(dcou))
                # st.write('prev model is  runs/train/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/train/exp'+str(dcou)
                current_model = 'runs/yolov5/val/exp'+str(dcou)
        else:
            if bbb == 1:
                # st.write('current model is  runs/train/exp'+str(dcou))
                # st.write('prev model is  runs/val/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/val/exp'
                current_model = 'runs/yolov5/train/exp'
            else:
                # st.write('current model is  runs/train/exp'+str(dcou))
                # st.write('prev model is  runs/val/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/val/exp'+str(dcou)
                current_model = 'runs/yolov5/train/exp'+str(dcou)
        
        df2 = pd.read_csv(current_model+'/metrics.csv')
        df1 = pd.read_csv(prev_best_model+'/metrics.csv')
        
        coll1 = df2["F1-Score"]
        coll1 = coll1.to_numpy()
        coll1 = np.reshape(coll1,(2,1))

        coll2 = df1["F1-Score"]
        coll2 = coll2.to_numpy()
        coll2 = np.reshape(coll2,(2,1))

        chart_data = pd.DataFrame(np.concatenate((coll1,coll2), axis = 1), columns=['best model', 'New model'])
        st.write("## F1-Score")
        st.line_chart(chart_data)
        col1, col2 = st.columns(2)
        col1.write("## New model")
        col1.write("### Confusion Matrix")
        col1.image(os.path.join(prev_best_model,"confusion_matrix.png"))
        col1.write('\n')
        col1.write("### F1 Curve")
        col1.image(os.path.join(prev_best_model,"F1_curve.png"))

        col2.write("## Previous best model")
        col2.write("### Confusion Matrix")
        col2.image(os.path.join(current_model,"confusion_matrix.png"))
        
        col2.write('\n')
        col2.write("### F1 Curve")
        col2.image(os.path.join(current_model,"F1_curve.png"))

        col1.write("### New model metrics")
        metrics_path = os.path.join(prev_best_model,"metrics.csv")
        df = pd.read_csv(metrics_path)
        col1.write(df)

        col2.write("### Previous best metrics")
        metrics_path = os.path.join(current_model,"metrics.csv")
        df = pd.read_csv(metrics_path)
        col2.write(df)
    else:
        st.title('TRAIN A DATASET TO EVALUATE METRICS')


def show_metrics_home():
    dcou = params["yolov5"]['ingest']['dcount']
    if dcou != 0:
        if params["yolov5"]["weights"] == "pretrained/best.pt":
            bbb = 1
        else:
            if params["yolov5"]["weights"].split("/")[3] == 'exp':
                bbb=1
            else:
                bbb = params["yolov5"]["weights"].split("/")[3]
                bbb = bbb[-1]

        if int(bbb) == int(dcou):
            if bbb == 1:
                # st.write('current model is  runs/val/exp')
                # st.write('prev model is  runs/train/exp')
                prev_best_model = 'runs/yolov5/train/exp'
                current_model = 'runs/yolov5/val/exp'
            else:
                # st.write('current model is  runs/val/exp'+str(dcou))
                # st.write('prev model is  runs/train/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/train/exp'+str(dcou)
                current_model = 'runs/yolov5/val/exp'+str(dcou)
        else:
            if bbb == 1:
                # st.write('current model is  runs/train/exp'+str(dcou))
                # st.write('prev model is  runs/val/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/val/exp'
                current_model = 'runs/yolov5/train/exp'
            else:
                # st.write('current model is  runs/train/exp'+str(dcou))
                # st.write('prev model is  runs/val/exp'+str(dcou))
                prev_best_model = 'runs/yolov5/val/exp'+str(dcou)
                current_model = 'runs/yolov5/train/exp'+str(dcou)
        
        df1 = pd.read_csv(current_model+'/metrics.csv')
        df2 = pd.read_csv(prev_best_model+'/metrics.csv')
        
        coll1 = df2["F1-Score"]
        coll1 = coll1.to_numpy()
        coll1 = np.reshape(coll1,(2,1))

        coll2 = df1["F1-Score"]
        coll2 = coll2.to_numpy()
        coll2 = np.reshape(coll2,(2,1))

        chart_data = pd.DataFrame(np.concatenate((coll1,coll2), axis = 1), columns=['New model', 'Previous model'])
        st.write("## F1-Score")
        st.line_chart(chart_data)
        col1, col2 = st.columns(2)
        col1.write("## New model")
        col1.write("### Confusion Matrix")
        col1.image(os.path.join(prev_best_model,"confusion_matrix.png"))
        col1.write('\n')
        col1.write("### F1 Curve")
        col1.image(os.path.join(prev_best_model,"F1_curve.png"))

        col2.write("## Previous model")
        col2.write("### Confusion Matrix")
        col2.image(os.path.join(current_model,"confusion_matrix.png"))
        
        col2.write('\n')
        col2.write("### F1 Curve")
        col2.image(os.path.join(current_model,"F1_curve.png"))

        col1.write("### New model metrics")
        metrics_path = os.path.join(prev_best_model,"metrics.csv")
        df = pd.read_csv(metrics_path)
        col1.write(df)

        col2.write("### Previous metrics")
        metrics_path = os.path.join(current_model,"metrics.csv")
        df = pd.read_csv(metrics_path)
        col2.write(df)
    else:
        st.title('TRAIN A DATASET TO EVALUATE METRICS')


def hero_page():
    st.image('screenshots/hero.jpeg', width=1000)

def predict_image():
    img = st.file_uploader("Upload Image")
    if img:
        with open(f'detect/image.png', "wb") as f:
            f.write(img.getbuffer())
        
        st.subheader('Predicting........')
        st.image('detect/image.png')
        detect.run(weights=params['yolov5']['weights'], source='detect/image.png')
        st.success('Prediction Successful')
        st.image('runs/yolov5/detect/exp/image.png')


def det_pipeline():
    st.subheader('Choose Dataset')
    opts = os.listdir('buffer')
    opts.sort()
    option = st.selectbox('',opts)
    params["detectron2"]['ingest']['dpath'] = option
    yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)
    
    if st.button('Run Pipeline'):
        st.subheader('Running Detectron2 Pipeline..........')
        shutil.rmtree('.dvc/cache', ignore_errors=True) 
        
        if not os.system("dvc repro"):
            st.success('Pipeline executed successfully')
            st.balloons()
            st.snow()
            det_metrics()
        else:
            st.error("Pipleine execution failed")


def det_predict():
    # st.header("Best Model {}".format(params['detectron2']['version']['best']))  
    best_model = params['detectron2']['version']['best']
    img = st.file_uploader("Upload Image")
    # path = 'archive/Images'
    os.makedirs("images", exist_ok = True)
    os.makedirs("detect", exist_ok = True)
    os.makedirs("runs/detectron2/detect/", exist_ok = True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["detectron2"]["hyps"]["config_file"]))

    if best_model == "v0" or params['detectron2']['weights'] == "pretrained":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['detectron2']['hyps']['config_file'])
    else:
        cfg.MODEL.WEIGHTS = os.path.join(params["detectron2"]["outputs"]["train_dir"],"exp{}".format((best_model)[1]),"weights/model_final.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["detectron2"]["hyps"]["NUM_CLASSES"]


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["detectron2"]["hyps"]["SCORE_THRESH_TEST"]
    predictor = DefaultPredictor(cfg)

    if best_model == "v0" or params['detectron2']['weights'] == "pretrained":
        st.header("Best Model is Detectron PreLoaded Weights")
    else:
        st.header("Best Model is {}".format(params['detectron2']['version']['best'])) 

    if img:
        with open(f'detect/image.png', "wb") as f:
            f.write(img.getbuffer())
        
        st.subheader('Predicting........')
        st.image('detect/image.png')
        img = cv2.imread('detect/image.png')
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
        cv2.imwrite("runs/detectron2/detect/image.png",out.get_image()[:, :, ::-1])
        st.success('Prediction Successful')
        st.image("runs/detectron2/detect/image.png")


def gen_confusion_matrix(tp,fp,fn,label):
    params = yaml.safe_load(open('params.yaml'))
    plt.clf()
    cm = [[0,fp[0]],[fn[0],tp[0]]]
    cm = np.array(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix for Pedestrian Detection')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

    if label == "new":
        plt.savefig(os.path.join(params["detectron2"]["outputs"]["train_dir"],"exp{}".format((params['detectron2']['ingest']['dcount']) - 1),"confusion_matrix.png"))
    else:
        plt.savefig(os.path.join(params["detectron2"]["outputs"]["val_dir"],"exp{}".format((params['detectron2']['ingest']['dcount']) - 1),"confusion_matrix.png"))


def det_metrics():

    params = yaml.safe_load(open('params.yaml'))
    # best_model = params['detectron2']['version']['best']
    if params['detectron2']['ingest']['dcount']-1 == 0 :
        st.title('TRAIN A DATASET TO EVALUATE METRICS')
    else:
        best_path = os.path.join(params["detectron2"]["outputs"]["val_dir"],"exp{}".format((params['detectron2']['ingest']['dcount']) - 1))
        new_path = os.path.join(params["detectron2"]["outputs"]["train_dir"],"exp{}".format((params['detectron2']['ingest']['dcount']) - 1))

        best_path_metrics = os.path.join(best_path, "predict_metrics.json")
        new_path_metrics = os.path.join(new_path, "predict_metrics.json")

        f1 = open (best_path_metrics, "r")
        metrics_best = json.loads(f1.read())

        f2 = open (new_path_metrics, "r")
        metrics_new = json.loads(f2.read())

        metrics_new_df = pd.DataFrame([metrics_new])
        metrics_best_df = pd.DataFrame([metrics_best])

        metrics_new_df = metrics_new_df.drop(['APm', 'APl'], axis=1)
        metrics_best_df = metrics_best_df.drop(['APm', 'APl'], axis=1)

        st.write("## Previous Best model")
        st.write(metrics_best_df)

        st.write("## New model")
        st.write(metrics_new_df)

        best = metrics_best_df["F1"][0]
        new = metrics_new_df["F1"][0]

        x = np.array(["previous best", "new"])
        y = np.array([best,new])
        plt.scatter(x, y, label = "set-1", color='r')
        plt.ylim(0,1)
        plt.xticks([-1,0,1,2])
        plt.xlabel("Models")
        plt.savefig(os.path.join(new_path,"F1.png"))
        st.write("### F1")
        st.image(os.path.join(new_path,"F1.png"))

        tp_new = metrics_new_df["tp"]
        fp_new = metrics_new_df["fp"]
        fn_new = metrics_new_df["fn"]

        tp_best = metrics_best_df["tp"]
        fp_best = metrics_best_df["fp"]
        fn_best = metrics_best_df["fn"]

        gen_confusion_matrix(tp_new,fp_new,fn_new, label = "new")
        gen_confusion_matrix(tp_best,fp_best,fn_best, label = "best")

        st.write("# Confusion Matrix")
        col1,col2 = st.columns(2)
        col1.write("### Previous Best model")
        col1.image(os.path.join(best_path,"confusion_matrix.png"))

        col2.write("### New model")
        col2.image(os.path.join(new_path,"confusion_matrix.png"))


def main():
    st.set_page_config(layout="wide")
    st.title("MLOps Pipeline for Pedestrian Detection")
    pages = {
        "Choose one of the following":hero_page,
        "Train Dataset": pipeline,
        "Predict on an Image": predict_image,
        "Metrics": show_metrics_home,
    }

    det_pages = {
        "Choose one of the following":hero_page,
        "Train Dataset": det_pipeline,
        "Predict on an Image": det_predict,
        "Metrics": det_metrics,
    }
    st.sidebar.image('screenshots/logo2.png')
    
    st.markdown("""---""")
    st.sidebar.markdown("""---""")
    st.sidebar.title('Select Model -')
    opp = st.sidebar.selectbox('',("Choose one of the following",'yolov5', 'detectron2'))

    if opp == "Choose one of the following":
        hero_page()

    elif opp == 'yolov5':
        params['model'] = 'yolov5'
        yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)
        selected_page = st.sidebar.selectbox('',pages.keys())
        pages[selected_page]()
    else:
        params['model'] = 'detectron2'
        yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)
        selected_page = st.sidebar.selectbox('',det_pages.keys())
        det_pages[selected_page]()

if __name__ == '__main__':
    main()