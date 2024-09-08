from os import listdir
from os.path import isfile, join
import os
import time
import yaml
import shutil

watchDirectory = 'buffer'
pollTime = 0

def filesInDirectory(my_dir: str):
    onlyfiles = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
    return(onlyfiles)

def runPipeline(newFiles: list):
    shutil.rmtree('.dvc/cache', ignore_errors=True) 
    print(f'Running Pipeline with {newFiles[0]}')

    params = yaml.safe_load(open('params.yaml'))

    if params['model'] == 'yolov5':
        params[params['model']]['ingest']['dcount'] = params[params['model']]['ingest']['dcount'] + 1
    
    params[params['model']]['ingest']['dpath'] = newFiles[0]

    yaml.dump(params, open('params.yaml', 'w'), sort_keys=False)

    if not os.system("dvc repro"):
        print('Pipeline executed successfully')
    else:
        print('Broken Pipeline')
        
def fileWatcher(my_dir: str, pollTime: int):
    print('Listening for new datasets....')
    while True:
        if 'watching' not in locals():
            previousFileList = filesInDirectory(watchDirectory)
            watching = 1
        
        time.sleep(pollTime)
        
        newFileList = filesInDirectory(watchDirectory)
        fileDiff = [x for x in newFileList if x not in previousFileList]
        previousFileList = newFileList
        if len(fileDiff) == 0: continue
        runPipeline(fileDiff)

fileWatcher(watchDirectory, pollTime)