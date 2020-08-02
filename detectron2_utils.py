"""
Import base libraries
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
import json

"""
Import Detectron Libraries
"""




# Function for loadig the val / train metrics for loss plotting
def load_json_arr(json_path):
    lines = []
    if json_path[-1] == "/":
        with open(os.path.join(json_path,"metrics.json"), "r") as f:
            for line in f:
                lines.append(json.loads(line))
    else:
        with open(os.path.join(json_path,"/metrics.json"),"r") as f:
            for line in f:
                lines.append(json.loads(line))
    return lines

# Function to plot the validation and training set
def plot_train_val_loss(path, figsize=(10,10)):

    data = load_json_arr(path)
    plt.figure(figsize=figsize)
    plt.title(label=f"{path.split('/')[-3]}\n{path.split('/')[-2]}")
    plt.plot([x["iteration"] for x in data],
             [x["total_loss"] for x in data],label="Training Loss")
    plt.plot([x["iteration"] for x in data if "validation_loss" in x],
             [x["validation_loss"] for x in data if "validation_loss" in x],label="Validation Loss")
    plt.legend(loc="upper right")
    plt.show()


def plot_boxplot(path,feature,figsize=(10,10)):


    assert feature in ['bbox/AP', 'bbox/AP50', 'bbox/AP75', 'bbox/APl', 'bbox/APm', 'bbox/APs', 'data_time', 'eta_seconds', 'fast_rcnn/cls_accuracy', 'fast_rcnn/false_negative', 'fast_rcnn/fg_cls_accuracy', 'iteration', 'loss_box_reg', 'loss_cls', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_loc', 'lr', 'mask_rcnn/accuracy', 'mask_rcnn/false_negative', 'mask_rcnn/false_positive', 'roi_head/num_bg_samples', 'roi_head/num_fg_samples', 'rpn/num_neg_anchors', 'rpn/num_pos_anchors', 'segm/AP', 'segm/AP50', 'segm/AP75', 'segm/APl', 'segm/APm', 'segm/APs', 'time', 'timetest', 'total_loss', 'validation_loss']

    data = load_json_arr(path)

    feature_tot = [adict[f"{feature}"] for adict in data if feature in adict.keys()]

    plt.figure(figsize=figsize)

    plt.title(label=f"Inspecting: {feature} from:\n{path.split('/')[-3]}\n{path.split('/')[-2]}")
    plt.boxplot(feature_tot)
    plt.show()


# Create the output_dir when saving the weight/log file
def create_output_dir(lr, batch_n):
    return f"lr={float(lr)}--batch_n={batch_n}_{int(time.time())}"


# Function for exporting the Model Configuration File
# for future usage or inference
def export_cfg(cfg,name,path):
    if "yaml" in name:
        with open(os.path.join(path,name),"w") as file:
            yaml.dump(cfg,file)
    else:
        with open(os.path.join(path,name+".yaml"),"w") as file:
            yaml.dump(cfg,file)


# Load image names (including path) of the internet sourced images
def get_internet_imgs():
    internet_names = os.listdir("/home/max/Desktop/dissertation/Mask_RCNN/barry_data/internet/")
    return ["/home/max/Desktop/dissertation/Mask_RCNN/barry_data/internet/"+file for file in internet_names]


# Extracting the LR and batch_size from the trained models filename
def get_batch_num(path):
    return float(path.split("=")[-1].split("_")[0])
def get_lr(path):
    return np.format_float_positional(float(path.split("=")[1].split("--")[0]))

