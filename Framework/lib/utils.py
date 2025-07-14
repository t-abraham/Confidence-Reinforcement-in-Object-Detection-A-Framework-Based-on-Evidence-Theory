# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:43:29 2023

@author: Tahasanul Abraham
"""
# %% Initialization of Libraries and Directory

# Standard library imports
import copy
import contextlib
import os
import platform
import time
import warnings

# Third-party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sn
from shapely import box
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

# Local application/library imports
from lib.roc_v2 import roc

# Global time variables for tic/toc functions
start_time = None
end_time = None

# --------------------------------------------------------------------
# Timing Functions
# --------------------------------------------------------------------
def tic():
    global start_time
    start_time = time.time()

def toc():
    global end_time
    end_time = time.time()

def time_duration():
    duration = end_time - start_time
    rounded_duration = round(duration, 1)
    if rounded_duration >= 3600:
        hours = rounded_duration // 3600
        minutes = (rounded_duration % 3600) // 60
        return f"{hours} hour(s) {minutes} minute(s)"
    elif rounded_duration >= 60:
        minutes = rounded_duration // 60
        return f"{minutes} minute(s)"
    else:
        return f"{rounded_duration} second(s)"

# --------------------------------------------------------------------
# Data Loader Helper
# --------------------------------------------------------------------
def collate_fn(batch):
    """
    Collate function to handle images and bounding boxes with varying sizes.
    """
    return tuple(zip(*batch))

# --------------------------------------------------------------------
# SQLAlchemy Database Checker
# --------------------------------------------------------------------
def sqlalchemy_db_checker(uri):
    engine = create_engine(uri)
    if not database_exists(engine.url):
        create_database(engine.url)
    del engine
    return uri

# --------------------------------------------------------------------
# Intersection Over Union (IoU) Calculation
# --------------------------------------------------------------------
def calculate_iou(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    box_1 = box(xmin=x1_min, ymin=y1_min, xmax=x1_max, ymax=y1_max)
    box_2 = box(xmin=x2_min, ymin=y2_min, xmax=x2_max, ymax=y2_max)
    
    if box_1.intersects(box_2):
        return box_1.intersection(box_2).area / box_1.union(box_2).area
    else:
        return 0

# --------------------------------------------------------------------
# ROC Group Formatter
# --------------------------------------------------------------------
# def roc_group_formater(all_preds_list, iou_threshold=0.5, score_threshold=0.5):    
#     all_detections = [{} for _ in range(len(all_preds_list))]
#     roc_groups = [{} for _ in range(len(all_preds_list))]
#     group_counter = 1
    
#     for all_preds_counter, pred in enumerate(all_preds_list):
#         for pred_counter, (pred_box, pred_label, pred_score) in enumerate(zip(pred["boxes"], pred["labels"], pred["scores"])):
#             all_detections[all_preds_counter]["prediction_{}".format(str(pred_counter+1).zfill(3))] = (
#                 round(float(pred_label.cpu().numpy())),
#                 [round(elem) for elem in pred_box.tolist()],
#                 float(pred_score.cpu().numpy())
#             )
            
#     for idx, working_all_detections in enumerate(all_detections):
#         while True:        
#             if not working_all_detections:
#                 break
#             current_keys = list(working_all_detections.keys())                
#             current_roc_group = []
#             keys_to_remove = []
            
#             bbox1 = copy.deepcopy(working_all_detections[current_keys[0]][1])
#             pre_copy = list(copy.deepcopy(working_all_detections[current_keys[0]]))
#             pre_copy.append(1)
#             current_roc_group.append(tuple(pre_copy))
#             keys_to_remove.append(current_keys[0])
            
#             for current_key in current_keys[1:]:
#                 bbox2 = copy.deepcopy(working_all_detections[current_key][1])
#                 score = copy.deepcopy(working_all_detections[current_key][2])
#                 iou = calculate_iou(
#                     x1_min=bbox1[0], 
#                     y1_min=bbox1[1], 
#                     x1_max=bbox1[2], 
#                     y1_max=bbox1[3], 
#                     x2_min=bbox2[0], 
#                     y2_min=bbox2[1], 
#                     x2_max=bbox2[2], 
#                     y2_max=bbox2[3],
#                 )
                
#                 if iou >= iou_threshold and score >= score_threshold:
#                     pre_copy = list(copy.deepcopy(working_all_detections[current_key]))
#                     pre_copy.append(iou)
#                     current_roc_group.append(tuple(pre_copy))
#                     keys_to_remove.append(current_key)
                    
#             for key in keys_to_remove:
#                 working_all_detections.pop(key)
                
#             roc_groups[idx]["Group_{}".format(str(group_counter).zfill(3))] = current_roc_group
#             group_counter += 1
    
#     # Remove groups with less than 2 detections
#     for group_dict in roc_groups:
#         for key in list(group_dict.keys()):
#             if len(group_dict[key]) < 2:
#                 group_dict.pop(key)
                
#     return roc_groups

def roc_group_formater(all_preds_list, score_threshold=None, iou_threshold=None):
    
    all_detections = [ { } for i in range( len( all_preds_list ) ) ]
    roc_groups = [ { } for i in range( len( all_preds_list ) ) ]
    group_box = [ { } for i in range( len( all_preds_list ) ) ]
    group_counter = 1
    
    for all_preds_counter, pred in enumerate(all_preds_list):
        for pred_counter, (pred_box, pred_label, pred_score) in enumerate( zip(pred["boxes"], pred["labels"], pred["scores"]) ):
            all_detections[all_preds_counter]["prediction_{}".format(str(pred_counter+1).zfill(3))] = (
                                                                                round( float( pred_label.cpu().numpy() ) ), 
                                                                                [ round(elem) for elem in pred_box.tolist() ],
                                                                                float( pred_score.cpu().numpy() )
                                                                                
                                                                            )
            
    for working_all_detections_counter, working_all_detections in enumerate(all_detections):
        while True:        
            info_left = len(working_all_detections.keys())
            current_roc_group = []
            keys_to_remove = []
            if info_left == 0:
                break
            current_keys = list(working_all_detections.keys())                
            
            bbox1 = copy.deepcopy( working_all_detections[current_keys[0]][1] )
            pre_copy = list( copy.deepcopy( working_all_detections[current_keys[0]] ) )
            pre_copy.append(1)
            pre_copy = tuple( pre_copy )
            current_roc_group.append( copy.deepcopy( pre_copy ) )
            keys_to_remove.append( current_keys[0] )
            
            for current_key in current_keys[1:]:
                bbox2 = copy.deepcopy( working_all_detections[current_key][1] )
                score = copy.deepcopy( working_all_detections[current_key][2] )
                iou = calculate_iou(
                                        x1_min = bbox1[0], 
                                        y1_min = bbox1[1], 
                                        x1_max = bbox1[2], 
                                        y1_max = bbox1[3], 
                                        
                                        x2_min = bbox2[0], 
                                        y2_min = bbox2[1], 
                                        x2_max = bbox2[2], 
                                        y2_max = bbox2[3],
                                    )
                
                if iou >= iou_threshold and score >= score_threshold:
                    pre_copy = list( copy.deepcopy( working_all_detections[current_key] ) )
                    # pre_copy[2] = pre_copy[2] * iou
                    pre_copy.append(iou)
                    pre_copy = tuple( pre_copy )
                    current_roc_group.append( copy.deepcopy( pre_copy ) )
                    keys_to_remove.append( current_key )
                    
            for key_to_remove in keys_to_remove:
                working_all_detections.pop( key_to_remove )
                
            roc_groups[working_all_detections_counter]["Group_{}".format(str(group_counter).zfill(3))] = current_roc_group
            group_box[working_all_detections_counter]["Group_{}".format(str(group_counter).zfill(3))] = None
            group_counter += 1
    
    for i in range( len( roc_groups ) ):
        for key in list(roc_groups[i].keys()):
            if len(roc_groups[i][key]) < 2:
                roc_groups[i].pop(key)
                group_box[i].pop(key)
            else:
                roc_groups[i][key], group_box[i][key]  = compute_group_box_and_ious(roc_groups[i][key])
                
    return roc_groups, group_box

def compute_group_box_and_ious(group):
    """
    group: list of (label, [x1, y1, x2, y2], score, prev_iou)
    Returns: new group where each tuple's last value is replaced with IoU(box, group_box)
    """
    boxes = [item[1] for item in group]
    scores = [item[2] for item in group]
    
    # Compute weighted group box
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32).reshape(-1, 1)
    weighted_sum = np.sum(boxes_np * scores_np, axis=0)
    total_weight = np.sum(scores_np)
    group_box = weighted_sum / total_weight
    
    # IoU function
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0
    
    # Replace prev_iou with actual IoU to group_box
    new_group = []
    for (g_label, g_box, g_score, _g_prev_iou) in group:
        box_iou = iou(g_box, group_box)
        new_group.append((g_label, g_box, g_score, box_iou))
    
    return new_group, group_box
    #return new_group

def process_roc(DEVICE, pred_keys, roc_groups, group_box, all_preds_list, deci=5, roc_regression=True, dual="false"):
    
    if str(dual).lower() == "true":   
        perform_roc = roc(deci, True)
    else:
        perform_roc = roc(deci, False)
    
    roc_preds = [ {j : [] for j in pred_keys} for i in range( len( all_preds_list ) ) ]
    
    for image_number in range( len( roc_groups ) ):
        for current_group in roc_groups[image_number].keys():
            classification_masses_roc = {}
            # xmin, ymin, xmax, ymax = 0, 0, 0, 0
            
            # if roc_regression is True:
            #     regression_masses_roc = {}
            # regression_masses_roc_total = 0
            for current_mass_id in range( len( roc_groups[image_number][current_group] ) ):
                if str(dual).lower() == "true":
                    classification_masses_roc["m{}".format(current_mass_id)] = \
                        {
                            roc_groups[image_number][current_group][current_mass_id][0]     :
                                [
                                    roc_groups[image_number][current_group][current_mass_id][2],
                                    roc_groups[image_number][current_group][current_mass_id][3]
                                ],
                        }
                elif str(dual).lower() == "iou":
                    classification_masses_roc["m{}".format(current_mass_id)] = \
                        {
                            roc_groups[image_number][current_group][current_mass_id][0]     :
                                roc_groups[image_number][current_group][current_mass_id][3],
                        }
                else:
                    classification_masses_roc["m{}".format(current_mass_id)] = \
                        {
                            roc_groups[image_number][current_group][current_mass_id][0]     :
                                roc_groups[image_number][current_group][current_mass_id][2],
                        }
                # if roc_regression is True:        
                #     regression_masses_roc["m{}".format(current_mass_id)] = \
                #         {
                #             str(roc_groups[image_number][current_group][current_mass_id][1])     :
                #                 roc_groups[image_number][current_group][current_mass_id][2] * roc_groups[image_number][current_group][current_mass_id][3],
                #         }
                #     regression_masses_roc_total += roc_groups[image_number][current_group][current_mass_id][2] * roc_groups[image_number][current_group][current_mass_id][3]
                
            roc_ds_classification_k, roc_ds_classification_matrix = perform_roc.perform_ds(**classification_masses_roc)
            
            # if roc_regression is True:
            #     for key in regression_masses_roc.keys():
            #         for bbox in regression_masses_roc[key].keys():
            #             regression_masses_roc[key][bbox] /= regression_masses_roc_total
                        
            #     xmin, ymin, xmax, ymax, bbox_normalizer = 0, 0, 0, 0, 0
                
            #     for key in regression_masses_roc.keys():
            #         for bbox in regression_masses_roc[key].keys():
            #             bbox_exploded = eval(bbox)
            #             xmin += bbox_exploded[0] * regression_masses_roc[key][bbox]
            #             ymin += bbox_exploded[1] * regression_masses_roc[key][bbox]
            #             xmax += bbox_exploded[2] * regression_masses_roc[key][bbox]
            #             ymax += bbox_exploded[3] * regression_masses_roc[key][bbox]
            #             bbox_normalizer += regression_masses_roc[key][bbox]
                
            #     xmin /= bbox_normalizer
            #     ymin /= bbox_normalizer
            #     xmax /= bbox_normalizer
            #     ymax /= bbox_normalizer
            # else:
            #     # Initialize max value and the element with max value
            #     max_value = float('-inf')
                
            #     # Iterate through the list
            #     for current_mass_id in range( len( roc_groups[image_number][current_group] ) ):
            #         # Check if the third element is greater than the current max
            #         if roc_groups[image_number][current_group][current_mass_id][2] > max_value:
            #             max_value = roc_groups[image_number][current_group][current_mass_id][2]
                        
            #             xmin, ymin, xmax, ymax = roc_groups[image_number][current_group][current_mass_id][1]
            
            roc_preds[image_number]["boxes"].append( [
                                                        round(group_box[image_number][current_group][0]), 
                                                        round(group_box[image_number][current_group][1]),
                                                        round(group_box[image_number][current_group][2]),
                                                        round(group_box[image_number][current_group][3])
                                                    ] )
            roc_preds[image_number]["labels"].append( max( roc_ds_classification_matrix, key=roc_ds_classification_matrix.get ) )
            roc_preds[image_number]["scores"].append( 1-roc_ds_classification_k )
        
        for key in list( roc_preds[image_number].keys() ):
            roc_preds[image_number][key] = torch.as_tensor(roc_preds[image_number][key])
            roc_preds[image_number][key] = roc_preds[image_number][key].to( DEVICE )
            
    return roc_preds

def emojis(string=''):
    MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string

# --------------------------------------------------------------------
# Helper Classes for Exception Handling
# --------------------------------------------------------------------
class TryExcept(contextlib.ContextDecorator):
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose
    def __enter__(self):
        pass
    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

# --------------------------------------------------------------------
# Metrics and Confusion Matrix
# --------------------------------------------------------------------
class metrics:
    """
    Class to compute Mean Average Precision and generate a confusion matrix.
    """
    def __init__(self, classes, score_thres=None, iou_thres=None, row=None, col=None):
        if "__background__" in classes:
            self.row = classes.index("__background__")
            self.col = classes.index("__background__")        
        else:
            self.row = row
            self.col = col
            
        self.classes = classes  
        self.score_thres = score_thres
        self.iou_thres = iou_thres        
        self.count_classes = len(self.classes)
        self.MeanAveragePrecision = MeanAveragePrecision(class_metrics=True)
        self.ConfusionMatrix = ConfusionMatrix(self.count_classes, self.score_thres, self.iou_thres, self.row, self.col)
        self.results = {}
    
    def _InputFormatter(self, loads):
        return_array = []
        for load in loads:
            if "scores" in load:
                for load_box, load_label, load_score in zip(load['boxes'], load['labels'], load['scores']):
                    if not torch.is_tensor(load_box):
                        load_box = torch.tensor(load_box)
                    if not torch.is_tensor(load_label):
                        load_label = torch.tensor(load_label)
                    if not torch.is_tensor(load_score):
                        load_score = torch.tensor(load_score)
                        
                    if load_box.dim() == 0:
                        load_box = torch.unsqueeze(load_box, 0)
                    if load_label.dim() == 0:
                        load_label = torch.unsqueeze(load_label, 0)
                    if load_score.dim() == 0:
                        load_score = torch.unsqueeze(load_score, 0)
                        
                    return_array.append(torch.cat((load_box, load_score, load_label)))
            else:
                for load_box, load_label in zip(load['boxes'], load['labels']):
                    if not torch.is_tensor(load_box):
                        load_box = torch.tensor(load_box)
                    if not torch.is_tensor(load_label):
                        load_label = torch.tensor(load_label)
                        
                    if load_box.dim() == 0:
                        load_box = torch.unsqueeze(load_box, 0)
                    if load_label.dim() == 0:
                        load_label = torch.unsqueeze(load_label, 0)
                        
                    return_array.append(torch.cat((load_label, load_box)))
        
        if return_array:
            return torch.stack(return_array)
        else:
            return torch.tensor([[]])
        
    def update(self, preds, targets):
        self.MeanAveragePrecision.update(preds, targets)
        formatted_preds = self._InputFormatter(preds)
        formatted_targets = self._InputFormatter(targets)
        
        if len(formatted_preds) and formatted_preds[0].numel() != 0:
            self.ConfusionMatrix.process_batch(formatted_preds, formatted_targets)
        
    def compute(self):
        self.results = self.MeanAveragePrecision.compute()
        self.results["confusion_matrix"] = self.ConfusionMatrix.get_matrix()
        self.results["tp_fp_fn"] = self.ConfusionMatrix.get_tp_fp_fn()
   
    def GetResults(self):
        return self.results
    
    def plot(self, key="", normalize=True, save_dir='', names=[]):
        self.ConfusionMatrix.cm_plot(key=key, normalize=normalize, save_dir=save_dir, arg_names=names)
    
    def print(self, key=""):
        if key:
            key = "({}) ".format(key)
            
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("********************************************************")
        print("----------------------------------------------------------------------")
        map_global = self.results["map"].numpy().tolist()
        if isinstance(map_global, (int, float)):
            map_global = [map_global]
        print("Model {} MAP Global: {}".format(key, [round(i, 3) for i in map_global]))
        print("----------------------------------------------------------------------")
        map_per_class = self.results["map_per_class"].numpy().tolist()
        if isinstance(map_per_class, (int, float)):
            map_per_class = [map_per_class]
        print("Model {} MAP per class: {}".format(key, [round(i, 3) for i in map_per_class]))
        print("----------------------------------------------------------------------")
        self.ConfusionMatrix.print()      
        print("----------------------------------------------------------------------")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("********************************************************")


class ConfusionMatrix:
    """
    Confusion matrix for object detection.
    """
    def __init__(self, nc, conf=0.25, iou_thres=0.5, row=None, col=None):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.row = row
        self.col = col
        
    def __box_iou__(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    
    def process_batch(self, detections, labels):
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = self.__box_iou__(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1

    def get_matrix(self):
        current_matrix = self.matrix
        if self.row is not None:
            current_matrix = np.delete(current_matrix, self.row, axis=0)
        if self.col is not None:
            current_matrix = np.delete(current_matrix, self.col, axis=1)
        return current_matrix

    def get_tp_fp_fn(self):
        current_matrix = self.matrix
        if self.row is not None:
            current_matrix = np.delete(current_matrix, self.row, axis=0)
        if self.col is not None:
            current_matrix = np.delete(current_matrix, self.col, axis=1)
        tp = current_matrix.diagonal()
        fp = current_matrix.sum(1) - tp
        fn = current_matrix.sum(0) - tp
        return tp[:-1], fp[:-1], fn[:-1]
    
    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def cm_plot(self, key="", normalize=True, save_dir='', arg_names=[]):
        if "__background__" in arg_names:
            names = copy.deepcopy(arg_names)
            names.remove("__background__")
        else:
            names = arg_names
        
        nc, nn = len(names), len(names)
            
        if key:
            key = "_{}".format(key)
        
        current_matrix = self.matrix
        if self.row is not None:
            current_matrix = np.delete(current_matrix, self.row, axis=0)
        if self.col is not None:
            current_matrix = np.delete(current_matrix, self.col, axis=1)
        
        array = current_matrix / ((current_matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)
        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if nc < 50 else 0.8)
        labels = (0 < nn < 99) and (nn == nc)
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(os.path.join(save_dir, f'confusion_matrix{key}.png'), dpi=250)
        plt.close(fig)

    def print(self):
        current_matrix = self.matrix
        if self.row is not None:
            current_matrix = np.delete(current_matrix, self.row, axis=0)
        if self.col is not None:
            current_matrix = np.delete(current_matrix, self.col, axis=1)
            
        for i in range(len(current_matrix)):
            print('; '.join(map(str, current_matrix[i])))
        tp, fp, fn = self.get_tp_fp_fn()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / ((1/precision) + (1/recall))
        print("Calculated:")
        print("True Positive: {}".format(tp))
        print("False Positive: {}".format(fp))
        print("False Negative: {}".format(fn))
        print("Calculated per Classes:")
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1 Score: {}".format(f1))
        print("Calculated Mean:")
        print("Precision: {}".format(precision.mean()))
        print("Recall: {}".format(recall.mean()))
        print("F1 Score: {}".format(f1.mean()))
