import os,sys
import yaml
from PIL import Image
import cv2
import shutil
import pandas as pd 
import glob
import os
from PIL import Image
import albumentations as A
from progress.bar import Bar
import extras.logger as logg

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/augment.py data/split data/augmented\n'
    )
    sys.exit(1)

#Global
count = 0

#AUGMENT THE DATA
def augmentation(image_path,annot_path,data,output_image_path,output_annot_path):
    bboxes = []
    prev_image = data['name'][0]
    bar = Bar('Processing', max=len(data))
    #bar = progressbar.ProgressBar(maxval = len(data)).start()
    for index,row in data.iterrows():
        row_image = row['name']
        if row_image == prev_image:

            #To store bboxes of respective image
            boxes = [row['x_center'],row['y_center'],row['width'],row['height'],row['class_id']]
            bboxes.append(boxes)
            
        else:
            #CHECK FOR VALUE ERRORS 
            try:
                aug_image, aug_boxes = transform(image_path,prev_image,bboxes)
                save_images(aug_image,aug_boxes,output_image_path,output_annot_path)
                
                #bar.update(index)
            except ValueError:
                pass 
            bboxes = []
            boxes = [row['x_center'],row['y_center'],row['width'],row['height'],row['class_id']]
            bboxes.append(boxes)
        prev_image = row_image
        bar.next()
    bar.finish()
    return 


#TRANSFORM THE DATA INTO CUSTOM DATA
def transform(img_path,image_name,bboxes):
    #To read in image from path
    image_name = image_name+'.jpg'
    path = os.path.join(img_path,image_name)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Augmentation Tranformation
    transform = A.Compose([
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.5),
    A.RandomSnow(p=0.2),
    # A.RandomFog(),
    # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3
    #            )
], bbox_params=A.BboxParams(format='yolo'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    return transformed_image, transformed_bboxes

#DRAW BOUNDING BOXES TO CHECK AND SAVE THE BOXES, LABELS IN TXT FORMAT
def save_images(aug_image,aug_boxes,output_image,output_annot):
    global count
    img_path = 'aug' + str(count) + '.jpg'
    annot_path = 'aug' + str(count) + '.txt'
    output_image_path= os.path.join(output_image,img_path)
    output_annot_path= os.path.join(output_annot,annot_path)
    f = open(output_annot_path,'w')
    for boxes in aug_boxes:
        f.write('{} {} {} {} {}\n'.format(boxes[4],boxes[0],boxes[1],boxes[2],boxes[3]))
        #cv2.rectangle(aug_image, (round(boxes[0]),round(boxes[1])), (round(boxes[0]+boxes[2]),round(boxes[1]+boxes[3])), (0,255,0), 3)
    aug = Image.fromarray(aug_image)
    count = count +1
    aug.save(output_image_path)
    return 

def convert_yolov5_to_dataFrame(annot_path):
    aug_list = []
    for files in sorted(glob.glob(str(annot_path+'/*.txt*'))):
        fileName = files.split('/')[-1].split('.')[0]
        with open(files, "r") as f:
            bbox = (f.read()).split('\n')
        for data in bbox[0:-1]:
            data = data.split()
            value = (
                fileName,
                int(data[0]),
                float(data[1]),
                float(data[2]),
                float(data[3]),
                float(data[4]),
            )
            aug_list.append(value)
    column_name = ['name','class_id', 'x_center', 'y_center', 'width', 'height']
    aug_df = pd.DataFrame(aug_list, columns = column_name)
    return aug_df

def copy_split_aug(data,path):
    for file in data:
        shutil.copy(file,path)

    return

def yolov5Model():
    image_path = os.path.join(sys.argv[1],"images","train")
    annot_path = os.path.join(sys.argv[1],"labels","train")
    #outputaug = os.path.join(sys.argv[2],f"v{params[params['model']]['ingest']['dcount']}")
    outputaug = sys.argv[2]
    os.makedirs(outputaug, exist_ok = True)
    output_image_path = os.path.join(outputaug,'images','train')
    output_annot_path = os.path.join(outputaug,'labels','train')
    os.makedirs(output_image_path,exist_ok= True)
    os.makedirs(output_annot_path,exist_ok=True)
    
    images = [os.path.join(image_path, x) for x in os.listdir(image_path) if x[-3:] == "jpg"]
    images.sort()
    print(images)
    labels = [os.path.join(annot_path, x) for x in os.listdir(annot_path) if x[-3:] == "txt"]
    labels.sort()
    copy_split_aug(images,output_image_path)
    copy_split_aug(labels,output_annot_path)
    
    txt_dataframe = convert_yolov5_to_dataFrame(annot_path)
    #Augmentations
    #Store augmentations in Train folder
    augmentation(image_path,annot_path,txt_dataframe,output_image_path,output_annot_path)

def detectron2():
    aug_path = os.path.join(sys.argv[2], f"v{params['detectron2']['ingest']['dcount']}")
    os.makedirs(aug_path, exist_ok = True)
    print("Augmented Building")

def main():
    logger.info('AUGMENTING')
    if params['model'] == 'yolov5':
        yolov5Model()
    elif params['model'] == 'detectron2':
        detectron2()
    logger.info('AUGMENTING COMPLETED')

if __name__ == "__main__":
    logger = logg.log("augment.py")
    params = yaml.safe_load(open('params.yaml'))
    
    main()