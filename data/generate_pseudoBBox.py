# use groundtuth segmentation task as pseudo bbox.

import json
from pycocotools import mask
import numpy as np
import cv2
from PIL import Image, ImageDraw




def get_gt_mask(gt_seg_dict, cxr_id, disease):
    gt_item = gt_seg_dict[cxr_id][disease]
    gt_mask = mask.decode(gt_item)
    scaled_mask = scale_mask(gt_mask, (320, 320))
    return scaled_mask

def scale_mask(mask, target_size):
    # Find the dimensions of the input mask
    h, w = mask.shape[:2]
    # Create a new mask of the target size
    scaled_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    # Calculate the scaling factor in both directions
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    # Calculate the new coordinates of the mask's contours
    contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour[:, :, 0] = (contour[:, :, 0] * scale_x).astype(np.int32)
        contour[:, :, 1] = (contour[:, :, 1] * scale_y).astype(np.int32)
    return scaled_mask




# 522 frontal gt masks in total in file: "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
# 279 frontal gt masks in total in file: "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_val.json"


if __name__ == '__main__':
    gt_seg_file_valid = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_val.json"
    disease_list = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    pseudo_masks = {}
    pseudo_bboxs = {}

    # create empty pseudo bboxs and masks
    for disease in disease_list:
        pseudo_masks[disease] = np.zeros((320, 320))
        pseudo_bboxs[disease] = np.zeros((0, 4))

    with open(gt_seg_file_valid) as json_file:
        gt_seg_dict = json.load(json_file)

    print(len(gt_seg_dict.keys()))  # 499 unique cxr ids in test set, 187 images in valid set

    for cxr_id in gt_seg_dict.keys():
        # print(cxr_id)
        if "lateral" in cxr_id:
            continue
        for disease in disease_list:
            # print(disease)
            gt_mask = get_gt_mask(gt_seg_dict, cxr_id, disease)
            if np.sum(gt_mask) != 0:
                pseudo_masks[disease] += gt_mask


    for disease in disease_list:
        # print(disease)
        pseudo_masks[disease] = np.where(pseudo_masks[disease] > 0, 1, 0).astype(int)

        x_min = np.where(pseudo_masks[disease] > 0)[1].min()
        y_min = np.where(pseudo_masks[disease] > 0)[0].min()
        x_max = np.where(pseudo_masks[disease] > 0)[1].max()
        y_max = np.where(pseudo_masks[disease] > 0)[0].max()

        pseudo_masks[disease] = pseudo_masks[disease].tolist()
        pseudo_bboxs[disease] = np.array([x_min, y_min, x_max-x_min, y_max-y_min]).astype(int).tolist() # save as x, y, w, h


    pseudo_masks_dict = "./pseudo_masks.json"
    pseudo_bboxs_dict = "./pseudo_bboxs.json"

    with open(pseudo_masks_dict, 'w') as json_file:
        json.dump(pseudo_masks, json_file)

    with open(pseudo_bboxs_dict, 'w') as json_file:
        json.dump(pseudo_bboxs, json_file)

    # save pseudo bboxs and masks
