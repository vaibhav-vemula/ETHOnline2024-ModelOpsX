import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml
import sys
import extras.logger as logg
import extras.xml_to_df as xml_convert
import extras.yolov5_converter as yolov5_format
import extras.logger as logg
# from helper.detectron_df import *

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/split.py data/prepared data/split\n'
    )
    sys.exit(1)

random.seed(108)

# Convert the info dict to the required yolo format and write it to disk

def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
def class_id_mapping():
    class_ids = params['yolov5']['class_id']
    name_to_id = {}
    for id in class_ids.keys():
        name_to_id[id] = class_ids[id]
    return name_to_id

def convert_and_save_annotations(class_name_to_id_mapping):
    input_path = sys.argv[1]
    annotations = [os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations') )if x[-3:] == "xml"]
    annotations.sort()
    image_path = os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images')
    annot_path = os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations')
    # Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = xml_convert.extract_info_from_xml(ann)
        #yolov5_format.convert_to_yolov5(info_dict,annot_path,class_name_to_id_mapping)
    return input_path

def get_img_annots(input_path):
    images = [os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images'))]
    annotations = [os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations')) if x[-3:] == "txt"]
    return images,annotations

def split_and_save(t_img,t_annot,v_img,v_annot):
    split_train_image_path = os.path.join(sys.argv[2],'images','train')
    split_train_annot_path = os.path.join(sys.argv[2],'labels','train')
    split_val_image_path = os.path.join(sys.argv[2],'images','val')
    split_val_annot_path = os.path.join(sys.argv[2],'labels','val')
    os.makedirs(split_train_annot_path,exist_ok=True)
    os.makedirs(split_train_image_path,exist_ok=True)
    os.makedirs(split_val_image_path,exist_ok=True)
    os.makedirs(split_val_annot_path,exist_ok=True)
    move_files_to_folder(t_img, split_train_image_path)
    move_files_to_folder(v_img, split_val_image_path)
    move_files_to_folder(t_annot, split_train_annot_path)
    move_files_to_folder(v_annot, split_val_annot_path)
    return

#Detectron 2 modules

def get_img_annots_det(input_path):
    images = [os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'images'))]
    annotations = [os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'annotations')) if x[-3:] == "xml"]
    return images,annotations

def split_and_save_det(t_img,t_annot,v_img,v_annot,outputsplit):
    split_train_image_path = os.path.join(outputsplit,'images','train')
    split_train_annot_path = os.path.join(outputsplit,'labels','train')
    split_val_image_path = os.path.join(outputsplit,'images','val')
    split_val_annot_path = os.path.join(outputsplit,'labels','val')
    os.makedirs(split_train_annot_path,exist_ok=True)
    os.makedirs(split_train_image_path,exist_ok=True)
    os.makedirs(split_val_image_path,exist_ok=True)
    os.makedirs(split_val_annot_path,exist_ok=True)
    move_files_to_folder(t_img, split_train_image_path)
    move_files_to_folder(v_img, split_val_image_path)
    move_files_to_folder(t_annot, split_train_annot_path)
    move_files_to_folder(v_annot, split_val_annot_path)
    return

def yolov5Model():
    class_name_to_id_mapping = class_id_mapping()
    #Get input Path with conversion
    input_path = sys.argv[1]
    #Get images and annotations path
    images, annotations = get_img_annots(input_path)
    images.sort()
    annotations.sort()
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = params['split']['val'], random_state = 1)
    split_and_save(train_images,train_annotations,val_images,val_annotations)

def detectron2():
    params = yaml.safe_load(open('params.yaml'))
    outputsplit = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    input_path = os.path.join(sys.argv[1])
    os.makedirs(outputsplit, exist_ok = True)
    images, annotations = get_img_annots_det(input_path)
    images.sort()
    annotations.sort()
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = params['split']['val'], random_state = 1)
    split_and_save_det(train_images,train_annotations,val_images,val_annotations,outputsplit)
    split_path = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    os.makedirs(split_path, exist_ok = True)
    annot_path_train = os.path.join(split_path,"labels","train")
    annot_path_val = os.path.join(split_path,"labels","val")
    train_output_annot = creatingInfoData(annot_path_train)
    df_train = train_output_annot
    df_val = creatingInfoData(annot_path_val)
    df_train.to_pickle(os.path.join(split_path,'v{}_train.pkl'.format(params['detectron2']['ingest']['dcount'])))
    df_val.to_pickle(os.path.join(split_path,'v{}_val.pkl'.format(params['detectron2']['ingest']['dcount'])))

def main():
    logger.info('SPLITTING')
    if params['model'] == 'yolov5':
        yolov5Model()
    elif params['model'] == 'detectron2':
        detectron2()
    logger.info('SPLITTING COMPLETED')

if __name__ == '__main__':
    logger = logg.log("split.py")
    params = yaml.safe_load(open('params.yaml'))
    main()