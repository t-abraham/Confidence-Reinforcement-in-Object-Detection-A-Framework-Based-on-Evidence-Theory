# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 21:57:21 2025

@author: Abraham
"""
# %% Initialization of Libraries and Directory

# Standard library imports
import inspect
import os
import sys
import datetime
import pathlib
import copy

# Add parent directories to sys.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, grandparentdir)

grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, grandgrandparentdir)

# Third-party imports
import torch
import yaml
import tqdm
import optuna

# Local application/library imports
from lib.model import create_model
from lib.utils import (
    tic,
    toc,
    collate_fn,
    time_duration,
    metrics,
    sqlalchemy_db_checker,
    roc_group_formater,
    process_roc,
)
from lib.dataloader_aio import ModelAllDataloader, ModelAllDataset


#%% System Configuration

config_path = os.path.join(currentdir, 'config.yaml')

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

all_classes = [
                    "__background__", "person" ,"bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat",
                   "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
             ]

dataset_dir = os.path.join(currentdir, "pascal_voc_2012")

model_count = None
model_position = "Top"
roc_regression = True

all_models_to_load = [
                        "vgg", 
                        "resnet", 
                        "convnext", 
                        "squeezenet", 
                        "efficientnet", 
                        "shufflenet", 
                        "mobilenet"
                    ]

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%% Setup Logger
def setup_logger( filename, runningdir ):

    import logging, builtins
        
    logs_path = pathlib.Path(runningdir).joinpath("logs")
    logs_path.mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(logs_path.joinpath(filename), 'w'))
        
    _print = print
    # builtins.print = lambda *tup : logger.info(str(" ".join([str(x) for x in tup])))
        
    # Modified lambda function to handle keyword arguments
    def custom_print(*args, **kwargs):
        if 'file' in kwargs and kwargs['file'] is not None:
            _print(*args, **kwargs)  # Calls the original print for specific file outputs
        else:
            logger.info(" ".join([str(x) for x in args]))
                
    builtins.print = custom_print
        
    return _print

#%% Load Models
def load_models():
    """
    Loads models from disk using torch.load() for a list of model names.
    
    Args:
        all_models_to_load (list): A list of model names to load.
        currentdir (str): The current directory path.
        all_classes (list or int): A list of classes or count of classes used for creating the model architecture.
        
    Returns:
        dict: A dictionary where keys are model names and values are the loaded model dictionaries.
    """
    all_models = {}
    for load_model_name in all_models_to_load:
        try:
            all_models[load_model_name] = {}  # Initialize an empty dict for this model
            model_path = os.path.join(currentdir, "models", load_model_name, f"{load_model_name}_model.pth")
            
            # Load the dictionary from the saved model file
            loaded_dict = torch.load(model_path, map_location=torch.device(DEVICE))
            
            # Extract 'model_state_dict' separately to avoid accidental overwrites
            model_state_dict = loaded_dict.pop("model_state_dict", None)
            
            # Merge the rest of the dictionary
            for key, value in loaded_dict.items():
                if key in all_models[load_model_name]:
                    print(f"Conflict detected for key '{key}', old value: {all_models[load_model_name][key]}, new value: {value}")
                all_models[load_model_name][key] = value
            
            # Clean up the temporary dictionary to free memory
            del loaded_dict
            
            # Create the model architecture (ensure create_model is defined elsewhere)
            all_models[load_model_name]['model'] = create_model(len(all_classes), True, load_model_name)
            
            # Load the state dictionary into the model, if available
            if model_state_dict:
                all_models[load_model_name]['model'].load_state_dict(model_state_dict)
            
            # Free up memory from model_state_dict
            del model_state_dict
            
            # Set the model to evaluation mode
            all_models[load_model_name]['model'].eval()
            
            print(f"Loaded information for Model: {load_model_name}")
            print(f"-> Epoch: {all_models[load_model_name]['epoch']}")
            print(f"-> mAP: {all_models[load_model_name]['map']}")
            print()
            
        except Exception as e:
            print(f"Error loading model '{load_model_name}': {e}")
            if load_model_name in all_models:
                del all_models[load_model_name]  # Remove the key if loading fails
                
    return all_models

#%% Data Inference

def data_inference(all_models, score_thres=0.25, iou_thres=0.50, roc_regression=True):
    
    if "__background__" in all_classes:
        idx = all_classes.index("__background__")
        row = idx
        col = idx
    else:
        row = None
        col = None
    
    record_metrics = {}
    
    record_metrics["ROC"] = metrics( all_classes, 0, 0, row, col )
    
    pbar = tqdm.tqdm(test_loader,total=len(test_loader), position=0, leave=True)
    for i, data in enumerate(pbar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        all_model_results = [ { } for i in range( len( images ) ) ]
        pred_keys = []
        
        for model_name in all_models.keys():            
            if not model_name in record_metrics:
                record_metrics[model_name] = metrics( all_classes, 0, 0, row, col )
            
            all_models[model_name]['model'].to( DEVICE )
            
            with torch.no_grad( ):
                preds = all_models[model_name]['model']( images )
                all_models[model_name]['model'].to("cpu")
        
            record_metrics[model_name].update( preds, targets )
        
            for pred in preds:
                for key in pred.keys():
                    if not key in pred_keys:
                        pred_keys.append(key)
                        
            for i, pred in enumerate(preds):
                if len( all_model_results[i] ) == 0:
                    for pred_key in pred.keys():
                        all_model_results[i][pred_key] = pred[pred_key]
                else:
                    for pred_key in pred.keys():
                        all_model_results[i][pred_key] = torch.cat( (all_model_results[i][pred_key], pred[pred_key]), 0 )
                        
        roc_groups = roc_group_formater( 
                                            all_preds_list = all_model_results, 
                                            score_threshold = score_thres, 
                                            iou_threshold = iou_thres 
                                        )
            
        roc_preds = process_roc( 
                                    DEVICE = DEVICE, 
                                    pred_keys = pred_keys, 
                                    roc_groups = roc_groups, 
                                    all_preds_list = all_model_results, 
                                    deci = 5,
                                    roc_regression = roc_regression
                                )
        
        record_metrics["ROC"].update( roc_preds, targets )       
        
    return record_metrics

#%% Process Data Inference
def process_data_inference(all_models, score_thres=0.25, iou_thres=0.50, trial_number=None, roc_regression=True):
    
    if trial_number is None:
        save_cm_dir = os.path.join(currentdir, 'CM', f'Top-{len(all_models)}-Models')
    else:
        save_cm_dir = os.path.join(currentdir, 'CM', f'Top-{len(all_models)}-Models', f'trail-{trial_number}')
    os.makedirs(save_cm_dir, exist_ok=True)
    
    inference_metrics = data_inference(all_models, score_thres, iou_thres, roc_regression)
    
    for key in inference_metrics.keys():
        inference_metrics[key].compute( )
        inference_metrics[key].print( key )
        inference_metrics[key].plot( key, True, save_cm_dir, all_classes )
        inference_metrics[key] = inference_metrics[key].GetResults( )
        
        inference_metrics[key]["map"] = inference_metrics[key]["map"].cpu().numpy().tolist()
        if not isinstance(inference_metrics[key]["map"], list):
            inference_metrics[key]["map"] = [inference_metrics[key]["map"]]  # Wrap the float in a list
            
        inference_metrics[key]["map_per_class"] = inference_metrics[key]["map_per_class"].numpy().tolist()
        if not isinstance(inference_metrics[key]["map_per_class"], list):
            inference_metrics[key]["map_per_class"] = [inference_metrics[key]["map_per_class"]]  # Wrap the float in a list
        
    return inference_metrics["ROC"]["map"]

#%% Optuna Run
class optuna_run:
    def __init__( self ):
        self.best_score = None
        self._best_score = float('-inf')
    
    def __call__( self, trial, all_models, roc_regression):
        
        # score_thres = trial.suggest_float("score_thres", 0, 1, step=0.1)
        # iou_thres = trial.suggest_float("iou_thres", 0, 1, step=0.1)
        score_thres = trial.suggest_float("score_thres", 0, 1, step=0.1)
        iou_thres = trial.suggest_float("iou_thres", 0, 1, step=0.1)
        
        print ( "**************************** Trial - {} ****************************".format( str( trial.number ) ) )
        print ("score_thres -> {}".format(score_thres))
        print ("iou_thres -> {}".format(iou_thres))
        print ( )
        print ( "*************************************************************" )
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        self._best_score = process_data_inference(
                                                    all_models, 
                                                    score_thres=score_thres, 
                                                    iou_thres=iou_thres, 
                                                    trial_number=trial.number, 
                                                    roc_regression=roc_regression
                                                )
        
        print(f"Best score for Trial - {trial.number} is {self.best_score}")
        
        return self._best_score
    
    def callback ( self, study, trial ):        
        if study.best_trial.number == trial.number:
            self.best_score = copy.deepcopy( self._best_score )

#%% Optuna Duplicate Iteration Pruner
class optuna_duplicate_iteration_pruner(optuna.pruners.BasePruner):
    """
    DuplicatePruner

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of parameters
    as a previously completed trial.
    """

    def prune( self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial" ) -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                print ( "********************* TRIAL - {} PRUNED *********************".format( str( trial.number ) ) )
                print ( )
                print ( "*************************************************************" )
                return True

        return False

#%% Optuna Stop When Trial Keep Being Pruned
class optuna_stop_when_trial_keep_being_pruned:
    
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()
            
#%% Get Models
def get_models(all_models, count=None, position='top'):
    # Sort the models by their 'map' value in descending order
    sorted_models = sorted(all_models.items(), key=lambda item: item[1]['map'], reverse=True)
    position = str(position).lower()
    
    if count is None:
        # Return all models in the desired order
        if position == 'top':
            return dict(sorted_models)
        elif position == 'bottom':
            return dict(sorted_models[::-1])
        else:
            raise ValueError("position must be either 'top' or 'bottom'")
    
    # Return a subset of models based on the position
    if position == 'top':
        selected_models = sorted_models[:count]
    elif position == 'bottom':
        # Select the bottom 'count' models, and reverse to list them from lowest to highest
        selected_models = sorted_models[-count:][::-1]
    else:
        raise ValueError("position must be either 'top' or 'bottom'")
    
    return dict(selected_models)
            
#%% Standalone Run
if __name__ == "__main__":
    
    _print = setup_logger( datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y"), currentdir )
    
    tic()
    all_models = get_models(load_models(), count=model_count, position=model_position)    
    toc()
    
#%% Dataloader        
    tic()
    
    all_datasets = ModelAllDataset(dataset_dir, all_classes)
    
    # train_data = all_datasets.get_training_data(all_classes, config)
    # train_loader = ModelAllDataloader(
    #                                     train_data, batch_size= config['trainer']['batch_size'],
    #                                     shuffle=config['trainer']['shuffle'],num_workers=config['trainer']['num_workers'],
    #                                     collate_fn = collate_fn
                                        
    #                                     )
        
    # val_data = all_datasets.get_validation_data(all_classes, config)
    # valid_loader = ModelAllDataloader(
    #                                     val_data, batch_size=config['validator']['batch_size'],  
    #                                     shuffle=config['validator']['shuffle'], num_workers=config['validator']['num_workers'], 
    #                                     collate_fn=collate_fn 
                                        
    #                                     )    
    
    test_data = all_datasets.get_testing_data(all_classes, config)
    test_loader = ModelAllDataloader(
                                        test_data, batch_size=config['infernce']['batch_size'],   
                                        shuffle=config['infernce']['shuffle'], num_workers=config['infernce']['num_workers'],
                                        collate_fn=collate_fn
                                        
                                        )
   
    toc()
    # print(f"Number of training samples: {len(train_data)}")    
    # print(f"Number of validation samples: {len(val_data)}\n")    
    print(f"Number of testing samples: {len(test_data)}\n")
    print(f"Data loading took : {time_duration()}\n")
    
    objective = optuna_run( )
    n_trials = 1
    study_name = f"Journal-3_{model_position}-{len(all_models)}-Models-Reg_{roc_regression}"
    
    storage_uri = "mysql://{}:{}@{}/{}".format( 
                                                    "optuna_study",
                                                    "optuna_study_pass",
                                                    "193.174.70.66:3306",
                                                    "tahasanul_optuna",
                                                )
    func = lambda trial: objective( 
                                        trial = trial,
                                        all_models = all_models, 
                                        roc_regression = roc_regression,
                                    )
    
    pruner = optuna_duplicate_iteration_pruner()
    stopper = optuna_stop_when_trial_keep_being_pruned(20)
    
    study = optuna.create_study( 
                                        study_name = study_name, 
                                        storage = sqlalchemy_db_checker( storage_uri ), 
                                        load_if_exists = True,
                                        directions = ["maximize"],
                                        pruner = pruner
                                    )
    study.optimize( 
                        func, 
                        n_trials=n_trials, 
                        callbacks=[
                                        objective.callback,
                                        optuna.study.MaxTrialsCallback( n_trials ),
                                        stopper,
                                   ], 
                        gc_after_trial=True, 
                        show_progress_bar=True,
                    )