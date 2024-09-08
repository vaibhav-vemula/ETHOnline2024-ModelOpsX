import os
import yaml
import zipfile
import sys
import extras.logger as logg

params = yaml.safe_load(open('params.yaml'))
model = params['model']
data_path = os.path.join('data', 'prepared', f"v{params[model]['ingest']['dcount']}")
datasets_path = os.path.join('datasets', f"v{params[model]['ingest']['dcount']}")

os.makedirs(data_path, exist_ok=True)
os.makedirs(datasets_path, exist_ok=True)
logger = logg.log("ingest.py")
logger.info('INGESTING DATASET')
sys.path.append('../')

with zipfile.ZipFile(f'buffer/{params[model]["ingest"]["dpath"]}',"r") as zipf:
    zipf.extractall(data_path)
    zipf.extractall(datasets_path)