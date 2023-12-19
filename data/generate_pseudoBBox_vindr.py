import json
from pycocotools import mask
import numpy as np
import cv2
from PIL import Image, ImageDraw
import pandas as pd
import os





def split_bbox_file(src_vindr_bbox_file, disease_list, num_of_pseudo_gudance_samples, dest_dir):
    src_df = pd.read_csv(src_vindr_bbox_file)
    selected_samples = {}
    for disease in disease_list:
        selected_samples[disease] = src_df[src_df["class_name"] == disease].sample(n=num_of_pseudo_gudance_samples, random_state=1)
    pseudo_gudance_df = pd.concat(selected_samples.values())
    remaining_df = src_df.drop(pseudo_gudance_df.index)
    pseudo_gudance_df = pseudo_gudance_df.reset_index(drop=True)
    remaining_df = remaining_df.reset_index(drop=True)
    pseudo_gudance_df.to_csv(os.path.join(dest_dir, 'pseudo_gudance_df.csv'))
    remaining_df.to_csv(os.path.join(dest_dir, 'remaining_df.csv'))





def compute_pseudo_guidance(pseudo_guidance_file, disease_list, img_size=320):
    # we train model with scaled image of size 320x320,
    # original images do not have fix size,
    # therefore need scale bbox according to original image size.

    pseudo_masks_dict = "./pseudo_masks_vindr.json"
    pseudo_bboxs_dict = "./pseudo_bboxs_vindr.json"
    weighted_pseudo_masks_dict = "./weighted_pseudo_masks_vindr.json"


    src_df = pd.read_csv(pseudo_guidance_file)

    pseudo_bboxs = {}
    pseudo_masks = {}
    weighted_pseudo_masks = {}

    # create empty pseudo bboxs and masks
    for disease in disease_list:
        # initailize pseudo bboxs to x_min=320(img size), y_min=320(img size), x_max=0, y_max=0
        # for convinent comparation with real bboxs
        pseudo_bboxs[disease] = np.array([img_size,img_size, 0, 0])
        pseudo_masks[disease] = np.zeros((img_size, img_size))
        weighted_pseudo_masks[disease] = np.zeros((img_size, img_size))

    for idx, row in src_df.iterrows():
        disease = row["class_name"]
        if disease in disease_list:
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])
            pseudo_masks[disease][y_min:y_max, x_min:x_max] += 1
            weighted_pseudo_masks[disease][y_min:y_max, x_min:x_max] += 1
            pseudo_bboxs[disease][0] = min(pseudo_bboxs[disease][0], x_min)
            pseudo_bboxs[disease][1] = min(pseudo_bboxs[disease][1], y_min)
            pseudo_bboxs[disease][2] = max(pseudo_bboxs[disease][2], x_max)
            pseudo_bboxs[disease][3] = max(pseudo_bboxs[disease][3], y_max)

    for disease in disease_list:
        # convert to x,y,w,h format to be consistent with my code
        pseudo_bboxs[disease][2] = pseudo_bboxs[disease][2] - pseudo_bboxs[disease][0]
        pseudo_bboxs[disease][3] = pseudo_bboxs[disease][3] - pseudo_bboxs[disease][1]
        pseudo_bboxs[disease] = pseudo_bboxs[disease].tolist()
        pseudo_masks[disease] = (np.where(pseudo_masks[disease] > 0, 1, 0).astype(int)).tolist()
        weighted_pseudo_masks[disease] = (weighted_pseudo_masks[disease] / np.max(weighted_pseudo_masks[disease])).tolist()

    with open(pseudo_masks_dict, 'w') as json_file:
        json.dump(pseudo_masks, json_file)

    with open(pseudo_bboxs_dict, 'w') as json_file:
        json.dump(pseudo_bboxs, json_file)

    with open(weighted_pseudo_masks_dict, 'w') as json_file:
        json.dump(weighted_pseudo_masks, json_file)




if __name__ == '__main__':

    disease_list = ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion']
    src_vindr_bbox_file = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train_resized.csv"
    num_of_pseudo_gudance_samples = 75
    dest_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/pseudo_guidance"


    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        split_bbox_file(src_vindr_bbox_file, disease_list, num_of_pseudo_gudance_samples, dest_dir)
        pseudo_guidance_df = pd.read_csv(os.path.join(dest_dir, "pseudo_gudance_df.csv"))
        remaining_df = pd.read_csv(os.path.join(dest_dir, "remaining_df.csv"))
    else:
        pseudo_guidance_df = pd.read_csv(os.path.join(dest_dir, "pseudo_gudance_df.csv"))
        remaining_df = pd.read_csv(os.path.join(dest_dir, "remaining_df.csv"))

    # src_df = pd.read_csv(src_vindr_bbox_file)
    # print("len(src_df): ",len(src_df))
    # print("len(pseudo_guidance_df): ",len(pseudo_guidance_df))
    # print("len(remaining_df): ",len(remaining_df))
    # print("len(pseudo_guidance_df) + len(remaining_df): ",len(pseudo_guidance_df) + len(remaining_df))

    compute_pseudo_guidance(os.path.join(dest_dir, "pseudo_gudance_df.csv"), disease_list, img_size=320)










