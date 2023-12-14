import os
import json
from pycocotools import mask
import numpy as np
import cv2
from PIL import Image, ImageDraw




def get_gt_mask(gt_seg_dict, cxr_id, disease, img_size):
    gt_item = gt_seg_dict[cxr_id][disease]
    gt_mask = mask.decode(gt_item)
    scaled_mask = scale_mask(gt_mask, (img_size, img_size))
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




def create_pseudoMask(gt_seg_file, disease_list, img_size=320, dest_dir="./"):
    pseudo_masks = {}
    pseudo_bboxs = {}
    weighted_pseudo_masks = {}

    pseudo_masks_dict_path = os.path.join(dest_dir, "pseudo_masks_chexpert.json")
    pseudo_bboxs_dict_path = os.path.join(dest_dir, "pseudo_bboxs_chexpert.json")
    weighted_pseudo_masks_dict_path = os.path.join(dest_dir, "weighted_pseudo_masks_chexpert.json")

    # create empty pseudo bboxs and masks
    for disease in disease_list:
        pseudo_masks[disease] = np.zeros((img_size, img_size))
        weighted_pseudo_masks[disease] = np.zeros((img_size, img_size))
        pseudo_bboxs[disease] = np.zeros((0, 4))

    with open(gt_seg_file) as json_file:
        gt_seg_dict = json.load(json_file)

    print(len(gt_seg_dict.keys()))  # 499 unique cxr ids in test set, 187 images in valid set

    for cxr_id in gt_seg_dict.keys():
        # print(cxr_id)
        if "lateral" in cxr_id:
            continue
        for disease in disease_list:
            # print(disease)
            gt_mask = get_gt_mask(gt_seg_dict, cxr_id, disease, img_size=320)
            if np.sum(gt_mask) != 0:
                pseudo_masks[disease] += gt_mask

    for disease in disease_list:
        # print(disease)
        weighted_pseudo_masks[disease] = (pseudo_masks[disease] / np.max(pseudo_masks[disease])).tolist()

        # weighted_mask = Image.fromarray((weighted_pseudo_masks[disease] * 255).astype(np.uint8))
        # weighted_mask.show()

        pseudo_masks[disease] = np.where(pseudo_masks[disease] > 0, 1, 0).astype(int)

        # BBmask = Image.fromarray((pseudo_masks[disease] * 255).astype(np.uint8))
        # BBmask.show()

        x_min = np.where(pseudo_masks[disease] > 0)[1].min()
        y_min = np.where(pseudo_masks[disease] > 0)[0].min()
        x_max = np.where(pseudo_masks[disease] > 0)[1].max()
        y_max = np.where(pseudo_masks[disease] > 0)[0].max()

        pseudo_masks[disease] = pseudo_masks[disease].tolist()
        pseudo_bboxs[disease] = np.array([x_min, y_min, x_max-x_min, y_max-y_min]).astype(int).tolist() # save as x, y, w, h

    # save pseudo bboxs and masks
    with open(weighted_pseudo_masks_dict_path, 'w') as json_file:
        json.dump(weighted_pseudo_masks, json_file)

    with open(pseudo_masks_dict_path, 'w') as json_file:
        json.dump(pseudo_masks, json_file)

    with open(pseudo_bboxs_dict_path, 'w') as json_file:
        json.dump(pseudo_bboxs, json_file)
















def write_out_masks(gt_seg_file, disease_list, img_size, dest_dir):
    if os.path.exists(dest_dir) == False:
        os.makedirs(dest_dir)

    with open(gt_seg_file) as json_file:
        gt_seg_dict = json.load(json_file)
    print(len(gt_seg_dict.keys()))  # 499 unique cxr ids in test set, 187 images in valid set

    img_list = []
    disease_count = {}
    for disease in disease_list:
        disease_count[disease] = 0

    for cxr_id in gt_seg_dict.keys():
        # print(cxr_id)
        if "lateral" in cxr_id:
            continue
        for disease in disease_list:
            # print(disease)
            gt_mask = get_gt_mask(gt_seg_dict, cxr_id, disease, img_size)
            if np.sum(gt_mask) != 0:
                img_list.append(cxr_id)
                out_path = os.path.join(dest_dir, cxr_id + "_" + disease + ".npy")
                np.save(out_path, gt_mask)

    img_list = list(set(img_list))
    file_name = os.path.join(dest_dir, "img_list.txt")
    with open(file_name, 'w') as f:
        for item in img_list:
            f.write("%s\n" % item)







# 522 frontal gt masks in total in file: "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
# 279 frontal gt masks in total in file: "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_val.json"

if __name__ == '__main__':
    current_task = "create_pseudo_masks" # select from the Tasks list below

    Tasks = ["create_pseudo_masks", "create_weighted_pseudo_masks", "write_out_masks_valid", "write_out_masks_test"]

    gt_seg_file_valid = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_val.json"
    gt_seg_file_test = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
    disease_list = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

    if current_task == "create_pseudo_masks":
        create_pseudoMask(gt_seg_file = gt_seg_file_valid, disease_list=disease_list, img_size=320, dest_dir="./")

    # if current_task == "create_weighted_pseudo_masks":
    #     create_weighted_pseudoMask(gt_seg_file = gt_seg_file_valid, disease_list=disease_list, img_size=320, dest_dir="./")

    if current_task == "write_out_masks_test":
        dest_dir = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/test_masks"
        write_out_masks(gt_seg_file = gt_seg_file_test, disease_list=disease_list, img_size= 320, dest_dir=dest_dir)

    if current_task == "write_out_masks_valid":
        dest_dir = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/valid_masks"
        write_out_masks(gt_seg_file = gt_seg_file_valid, disease_list=disease_list, img_size= 320, dest_dir=dest_dir)






