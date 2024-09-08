import os,json

def results_logger(det_metrics,metrics,path,params):
    
    hyper_parameters = {
        "config_file" : params['detectron2']['hyps']['config_file'],
        "NUM_WORKERS" : params['detectron2']['hyps']['NUM_WORKERS'],
        "IMS_PER_BATCH" : params['detectron2']['hyps']['IMS_PER_BATCH'],
        "BASE_LR" : params['detectron2']['hyps']['BASE_LR'],
        "WARM_UP_ITERS" :params['detectron2']['hyps']['WARM_UP_ITERS'],
        "MAX_ITER" :params['detectron2']['hyps']['MAX_ITER'],
        "GAMMA" :params['detectron2']['hyps']['GAMMA'],
        "BATCH_SIZE_PER_IMAGE" : params['detectron2']['hyps']['BATCH_SIZE_PER_IMAGE'],
        "NUM_CLASSES" : params['detectron2']['hyps']['NUM_CLASSES'],
        "EVAL_PERIOD" : params['detectron2']['hyps']['EVAL_PERIOD'],
        "SCORE_THRESH_TEST" : params['detectron2']['hyps']['SCORE_THRESH_TEST']
    }

    my_metrics = {
        "tp" : metrics["TP"],
        "fp" : metrics["FP"],
        "fn" : metrics["FN"],
        "Precision" : metrics["Precision"],
        "Recall" : metrics["Recall"],
        "F1" : metrics["F1"],
        "Avg_IOU" : metrics["Avg_IOU"],
        'AP': det_metrics["AP"],
        'AP50': det_metrics["AP50"],
        'AP75': det_metrics["AP75"],
        'APs': det_metrics['APs'],
        'APm': det_metrics['APm'],
        'APl': det_metrics['APl']
    }


    with open(os.path.join(path,'det2_hyperparamters.json'), 'w') as fout:
        fout.write(json.dumps(hyper_parameters, indent = len(hyper_parameters)))

    with open(os.path.join(path,'predict_metrics.json'), 'w') as fout:
        fout.write(json.dumps(my_metrics, indent = len(my_metrics)))