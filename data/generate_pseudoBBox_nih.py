import json
from pycocotools import mask
import numpy as np
import cv2
from PIL import Image, ImageDraw
import pandas as pd
import os




def split_bbox_file(original_nih_bbox_file, ratio = 0.4):
    out_dir = os.path.dirname(original_nih_bbox_file)
    src_df = pd.read_csv(original_nih_bbox_file)
    img_list = src_df["Image Index"].unique().tolist()
    np.random.shuffle(img_list)
    split_idx = int(len(img_list) * ratio)
    valid_list = img_list[:split_idx]
    test_list = img_list[split_idx:]
    valid_df = src_df[src_df["Image Index"].isin(valid_list)]
    test_df = src_df[src_df["Image Index"].isin(test_list)]
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    valid_df.to_csv(os.path.join(out_dir, 'BBox_valid_df.csv'))
    test_df.to_csv(os.path.join(out_dir, 'BBox_test_df.csv'))


def compute_pseudo_bbox(src_nih_bbox_file, src_img_dir, disease_list, img_size= 320):
    # we train model with scaled image of size 320x320,
    # original images do not have fix size,
    # therefore need scale bbox according to original image size.

    src_df = pd.read_csv(src_nih_bbox_file)
    disease_stats = np.unique(np.array(src_df["Finding Label"].tolist()), return_counts=True)
    print(disease_stats)
    '''
    (array(['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax'], dtype='<U12'), array([76, 54, 52, 56, 33, 39, 49, 37]))
    '''

    pseudo_bboxs = {}
    pseudo_masks = {}
    # create empty pseudo bboxs and masks
    for disease in disease_list:
        # initailize pseudo bboxs to x_min=320(img size), y_min=320(img size), x_max=0, y_max=0
        # for convinent comparation with real bboxs
        pseudo_bboxs[disease] = np.array([img_size,img_size,0,0])
        pseudo_masks[disease] = np.zeros((img_size, img_size))

    for idx, row in src_df.iterrows():
        img_path = os.path.join(src_img_dir, src_df.iloc[idx]['Image Index'])
        img = Image.open(img_path).convert("L")
        disease = row["Finding Label"]
        if disease in disease_list:

            scale_factor_x = img_size / float(img.size[0])
            scale_factor_y = img_size / float(img.size[1])

            x_min = int(row["Bbox [x"] * scale_factor_x)
            y_min = int(row["y"]* scale_factor_y)
            x_max = int(x_min + row["w"] * scale_factor_x)
            y_max = int (y_min + row["h]"] * scale_factor_y)

            pseudo_masks[disease][y_min:y_max, x_min:x_max] += 1

            pseudo_bboxs[disease][0] = min(pseudo_bboxs[disease][0], x_min)
            pseudo_bboxs[disease][1] = min(pseudo_bboxs[disease][1], y_min)
            pseudo_bboxs[disease][2] = max(pseudo_bboxs[disease][2], x_max)
            pseudo_bboxs[disease][3] = max(pseudo_bboxs[disease][3], y_max)

    for disease in disease_list:
        # convert to x,y,w,h format to be consistent with my code
        pseudo_bboxs[disease][2] = pseudo_bboxs[disease][2] - pseudo_bboxs[disease][0]
        pseudo_bboxs[disease][3] = pseudo_bboxs[disease][3] - pseudo_bboxs[disease][1]

    return pseudo_bboxs, pseudo_masks





if __name__ == '__main__':
    src_nih_bbox_file = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_valid_df.csv"
    src_img_dir = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/images"
    if not os.path.exists(src_nih_bbox_file):
        original_nih_bbox_file = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_List_2017.csv"
        split_bbox_file(original_nih_bbox_file, ratio=0.4)

    disease_list = ['Cardiomegaly', 'Atelectasis', 'Effusion'] # overlap only three diseases
    pseudo_bboxs, pseudo_masks = compute_pseudo_bbox(src_nih_bbox_file, src_img_dir, disease_list, img_size= 320)




    union_mask = np.zeros((320, 320))
    for disease in disease_list:
        # print(disease, pseudo_bboxs[disease])
        # array = np.zeros((320, 320))
        # array[int(pseudo_bboxs[disease][1]):int(pseudo_bboxs[disease][1] + pseudo_bboxs[disease][3]), int(pseudo_bboxs[disease][0]):int(pseudo_bboxs[disease][0] + pseudo_bboxs[disease][2])] = 1
        # img = Image.fromarray((array * 255).astype(np.uint8))
        # img.show()
        # pseudo_masks[disease] = array.tolist()

        pseudo_masks[disease] = np.where(pseudo_masks[disease] > 0, 1, 0).astype(int)
        union_mask += pseudo_masks[disease]

        BBmask = Image.fromarray((pseudo_masks[disease] * 255).astype(np.uint8))
        BBmask.show()

        pseudo_bboxs[disease] = pseudo_bboxs[disease].tolist()
        pseudo_masks[disease] = pseudo_masks[disease].tolist()

    union_mask = np.where(union_mask > 0, 1, 0).astype(int)
    union_bbox = np.array([np.min(np.nonzero(union_mask)[1]), np.min(np.nonzero(union_mask)[0]), np.max(np.nonzero(union_mask)[1]), np.max(np.nonzero(union_mask)[0])])
    for disease in ['Consolidation', 'Edema']:
        pseudo_bboxs[disease] = union_bbox
        pseudo_masks[disease] = union_mask
        pseudo_bboxs[disease] = pseudo_bboxs[disease].tolist()
        pseudo_masks[disease] = pseudo_masks[disease].tolist()


    pseudo_masks_dict = "./pseudo_masks_nih.json"
    pseudo_bboxs_dict = "./pseudo_bboxs_nih.json"





    with open(pseudo_masks_dict, 'w') as json_file:
        json.dump(pseudo_masks, json_file)

    with open(pseudo_bboxs_dict, 'w') as json_file:
        json.dump(pseudo_bboxs, json_file)

    # save pseudo bboxs and masks
