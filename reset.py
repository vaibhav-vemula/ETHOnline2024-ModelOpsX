import yaml
import sys

params = yaml.safe_load(open('params.yaml'))
model = sys.argv[1]

if model == "yolo":
    print('---------CONFIGURING params.yaml for YOLOV5----------')
    params['yolov5']['ingest']['dcount'] = 0
    params['yolov5']['weights'] = 'pretrained/best.pt'
    params['yolov5']['hyps']['epochs'] = 1
    print('---------CONFIGURING params.yaml for YOLOV5 DONEEEE----------')
elif model == "det":
    print('---------CONFIGURING params.yaml for detectron2----------')
    params['detectron2']['ingest']['dcount'] = 1
    params['detectron2']['weights'] = 'pretrained'
    params['detectron2']['version']['best'] = 'v0'
    print('---------CONFIGURING params.yaml for detectron2 DONEEEE----------')
else:
    exit(0)

yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)
