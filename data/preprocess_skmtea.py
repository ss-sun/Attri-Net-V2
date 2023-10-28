import json
import csv
import pandas as pd
import numpy as np
import os
import torch
from typing import Union, Sequence
import matplotlib.pyplot as plt
import glob
import h5py
import pandas as pd
from tqdm import tqdm


def create_csv_files(src_imgs_dir, src_bbox_masks_dir, src_anno_json, dest_csv):
    with open(src_anno_json, "r") as json_file:
        # Parse the JSON content into a Python dictionary
        data = json.load(json_file)
    categories = data["categories"]
    id2name = {}
    id2superid = {}
    superid2supername = {}

    for cate in categories:
        supercategory = cate["supercategory"]
        supercategory_id = cate["supercategory_id"]
        id = cate["id"]
        name = cate["name"]
        id2name[id] = name
        id2superid[id] = supercategory_id
        superid2supername[supercategory_id] = supercategory
    # 1. create a csv file for each supercategory
    # Create an empty DataFrame with column names [image_id, Meniscal Tear, Ligament Tear, Cartilage Lesion, Effusion]
    column_names = ["image_id"]
    for i in range(len(superid2supername.keys())):
        disease = superid2supername[i + 1]
        column_names.append(disease)
    df = pd.DataFrame(columns=column_names)

    # fill in df
    scans = data["images"]
    for scan in scans:
        scan_name = scan["file_name"][:-3]
        scan_dir = os.path.join(src_imgs_dir, scan_name)
        slice_list = os.listdir(scan_dir)
        if len(slice_list) != 160:
            print("scan {} has {} slices!".format(scan_name, len(slice_list)))
        mask_dir = os.path.join(src_bbox_masks_dir, scan_name)
        mask_list = os.listdir(mask_dir)
        if len(mask_list) == 0:
            for index in tqdm(range(len(slice_list))):
                slice = slice_list[index]
                img_id = scan_name + '_' + slice[:-4]
                row_data = {key: None for key in column_names}
                row_data["image_id"] = img_id
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            for mask in mask_list:
                mask_id = mask[:-4]
                disease = superid2supername[id2superid[int(mask_id)]]
                gt_bbox_mask = np.load(os.path.join(mask_dir, mask))

                for index in tqdm(range(len(slice_list))):
                    slice = slice_list[index]
                    img_id = scan_name + '_' + slice[:-4]
                    slice_bbox_mask = gt_bbox_mask[:, :, index]
                    if np.sum(slice_bbox_mask) > 0:
                        lbl = 1
                    else:
                        lbl = 0

                    row_idx = df.index[df['image_id'] == img_id].tolist()
                    if len(row_idx) == 0:
                        # create a new row
                        row_data = {key: None for key in column_names}
                        row_data["image_id"] = img_id
                        row_data[disease] = lbl
                        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                    else:
                        df.at[row_idx[0], disease] = lbl

    # save the DataFrame to a csv file
    df_filled = df.fillna(0)
    df_filled.to_csv(dest_csv, index=False)






def fill_mask(bbox, empty_mask):
    # The bounding box (bbox) format is
    # [top left X position, top left Y position, top left Z position,
    # deltaX, deltaY, deltaZ].
    # NOTE: X corresponds to row (SI), Y to column (AP), and Z (RL/LR) to depth.
    x, y, z, w, h, d = bbox
    x = int(x)
    y = int(y)
    z = int(z)
    w = int(w)
    h = int(h)
    d = int(d)
    empty_mask[y:y + h, x:x + w, z:z + d] = 1
    return empty_mask

def create_bbox_masks(src_file, dest_folder):
    with open(src_file, "r") as json_file:
        # Parse the JSON content into a Python dictionary
        data = json.load(json_file)

    images = data["images"]
    # scan_names = []
    # scan_ids = []
    scans = {}
    for img in images:
        id = img["id"]
        file_name = img["file_name"]
        matrix_shape = img["matrix_shape"]
        folder_name = file_name[:-3]
        os.makedirs(os.path.join(dest_folder, folder_name), exist_ok=True)
        # scan_names.append(folder_name)
        # scan_ids.append(id)
        scans[id] = {"scan_name": folder_name, "matrix_shape": matrix_shape}

    annotations = data["annotations"]
    for anno in annotations:
        image_id = anno["image_id"]
        scan_name = scans[image_id]["scan_name"]
        matrix_shape = scans[image_id]["matrix_shape"]
        category_id = anno["category_id"]
        bbox = anno["bbox"]
        empty_mask = np.zeros(tuple(matrix_shape))
        bbox_mask = fill_mask(bbox, empty_mask)
        file_path = os.path.join(dest_folder, scan_name, f"{category_id:02d}" + ".npy")
        np.save(file_path, bbox_mask)



def write_images(src_dir, dest_dir):
    # Use the glob module to find all .h5 files in the directory
    h5_files = glob.glob(os.path.join(src_dir, '*.h5'))

    # Print the list of .h5 files
    for file in h5_files:
        print(file)
        scan_name = os.path.basename(file)[:-3]
        print(scan_name)
        os.makedirs(os.path.join(dest_dir, scan_name), exist_ok=True)
        with h5py.File(file, 'r') as data:
            # Get T1 and T2 images
            target = data['target']
            # print(target.shape)  # output: (512, 512, 160, 2, 1)
            num_slices = target.shape[2]
            for i in range(num_slices):
                slice = target[:, :, i, 0, 0]
                # print(slice.shape)  # output: (512, 512)
                abs_slice = np.abs(slice)
                file_name = os.path.join(dest_dir, scan_name, f"{i:03d}" + ".npy")
                np.save(file_name, abs_slice)
            # scale_img = get_scaled_image(abs_slice, 0.95, clip=True)
            # print(np.max(scale_img)) # output: 1.0
            # print(np.min(scale_img)) # output: 0.00015
            # plt.imshow(abs_slice, cmap='gray')
            # plt.show()




if __name__ == '__main__':

    '''
    # # write images
    # src_dir = "/mnt/qb/baumgartner/rawdata/SKM-TEA/skm-tea/v1-release/files_recon_calib-24"
    # dest_dir = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_imgs"
    # write_images(src_dir, dest_dir)
    
    '''

    '''
    # create ground truth bbox masks
    src_root = "/mnt/qb/work/baumgartner/sun22/data/skm-tea"
    dest_root = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox"
    for dataset in ["train", "val", "test"]:
        dest_dir = os.path.join(dest_root, dataset)
        src_json = os.path.join(src_root, "annotations", "v1.0.0", dataset +".json")
        create_bbox_masks(src_json, dest_dir)
    '''

    '''
    # create csv files
    src_imgs_dir = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_imgs"
    # src_bbox_masks_dir = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/train"
    src_root = "/mnt/qb/work/baumgartner/sun22/data/skm-tea"
    for dataset in ["train", "val", "test"]:
        src_bbox_masks_dir = os.path.join("/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/", dataset)
        src_anno_json = os.path.join(src_root, "annotations", "v1.0.0", dataset +".json" )
        dest_csv = os.path.join("/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/", dataset + ".csv")
        create_csv_files(src_imgs_dir, src_bbox_masks_dir, src_anno_json, dest_csv)

    '''

