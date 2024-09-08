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
import extras.logger as logg
# from extras.detectron_df import *

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/transform.py data/prepared data/transform\n'
    )
    sys.exit(1)

random.seed(108)

# Convert the info dict to the required yolo format and write it to disk

import os
#['id',x_center,y_center,width,height]


def convert_to_yolov5(info_dict,annot_path,class_name_to_id_mapping):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
    # Name of the file which we have to save 
    save_file_name = os.path.join(annot_path, info_dict["filename"].replace("xml", "txt"))
    print(save_file_name)
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))

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
    images = [os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images') )if x[-3:] == "jpg"]
    images.sort()

    output_path = sys.argv[2]
    image_path = os.path.join(output_path,f"v{params['yolov5']['ingest']['dcount']}",'images')
    annot_path = os.path.join(output_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations')
    os.makedirs(image_path,exist_ok=True)
    os.makedirs(annot_path,exist_ok=True)
    for image in images:
        shutil.copy(image,image_path)
    # Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = xml_convert.extract_info_from_xml(ann)
        convert_to_yolov5(info_dict,annot_path,class_name_to_id_mapping)
    return input_path

def get_img_annots(input_path):
    images = [os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'images'))]
    annotations = [os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['yolov5']['ingest']['dcount']}",'annotations')) if x[-3:] == "txt"]
    return images,annotations


#Detectron 2 modules

def get_img_annots_det(input_path):
    images = [os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'images', x) for x in os.listdir(os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'images'))]
    annotations = [os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'annotations', x) for x in os.listdir(os.path.join(input_path,f"v{params['detectron2']['ingest']['dcount']}",'annotations')) if x[-3:] == "xml"]
    return images,annotations


def copy_data(input,output):
    for file in input:
        shutil.copy(file,output)
    return


def yolov5Model():
    class_name_to_id_mapping = class_id_mapping()
    #Get input Path with conversion
    input_path = convert_and_save_annotations(class_name_to_id_mapping)

def detectron2():
    params = yaml.safe_load(open('params.yaml'))
    image_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}","images")
    annots_path = os.path.join(sys.argv[1],f"v{params['detectron2']['ingest']['dcount']}","annotations")
    
    output_image_path = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}","images")
    output_annot_path = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}","annotations")

    images = [os.path.join(image_path, x) for x in os.listdir(image_path )if x[-3:] == "jpg"]
    images.sort()
    annots = [os.path.join(annots_path, x) for x in os.listdir(annots_path) if x[-3:] == "xml"]
    annots.sort()
    
    os.makedirs(output_image_path, exist_ok = True)
    os.makedirs(output_annot_path, exist_ok = True)


    copy_data(images,output_image_path)
    copy_data(annots,output_annot_path)
    print("Transform building")
    # input_path = os.path.join(sys.argv[1])
    # os.makedirs(outputsplit, exist_ok = True)
    # images, annotations = get_img_annots_det(input_path)
    # images.sort()
    # annotations.sort()
    # train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = params['split']['val'], random_state = 1)
    # #split_and_save_det(train_images,train_annotations,val_images,val_annotations,outputsplit)
    # split_path = os.path.join(sys.argv[2],f"v{params['detectron2']['ingest']['dcount']}")
    # os.makedirs(split_path, exist_ok = True)
    # annot_path_train = os.path.join(split_path,"labels","train")
    # annot_path_val = os.path.join(split_path,"labels","val")
    # train_output_annot = creatingInfoData(annot_path_train)
    # df_train = train_output_annot
    # df_val = creatingInfoData(annot_path_val)
    # df_train.to_pickle(os.path.join(split_path,'v{}_train.pkl'.format(params['detectron2']['ingest']['dcount'])))
    # df_val.to_pickle(os.path.join(split_path,'v{}_val.pkl'.format(params['detectron2']['ingest']['dcount'])))

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