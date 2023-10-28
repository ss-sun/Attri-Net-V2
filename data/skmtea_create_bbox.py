import numpy as np
import pandas as pd
import os

# this file create BBox data for test set of SKM-TEA dataset
# input: test.csv("/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test.csv"), BBox masks( "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test")
# output: test_BBox.csv

test_csv = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test.csv"
bbox_dir = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test"
dest_csv = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test_BBox.csv"
superIDtoid = {
            1:[1,2,3,4,5,6,7,8],
            2:[9,10,11],
            3:[12,13,14,15],
            4:[16]
} #disease supercategory to catergory

idtosuperID = {
    1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,
    9:2,10:2,11:2,
    12:3,13:3,14:3,15:3,
    16:4
} #disease catergory to supercategory

TRAIN_DISEASES = ["Meniscal Tear", "Ligament Tear", "Cartilage Lesion", "Effusion"]

source_df = pd.read_csv(test_csv)
column_names = ['Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]']
bbox_df = pd.DataFrame(columns=column_names)


scans_list = os.listdir(bbox_dir)
for scan_id in scans_list:
    masks_list = os.listdir(os.path.join(bbox_dir, scan_id))
    # print(scan_id)
    for mask_file in masks_list:
        # load BBox mask
        BBox_mask = np.load(os.path.join(bbox_dir, scan_id, mask_file))
        disease_idx = idtosuperID[int(mask_file[:-4])]-1
        disease = TRAIN_DISEASES[disease_idx]

        for slice_id in range(BBox_mask.shape[2]):
            img_id = scan_id + '_' + f"{slice_id:03d}"
            lbls = source_df.loc[source_df['image_id'] == img_id][TRAIN_DISEASES].values.tolist()
            lbls = np.array(lbls).squeeze()
            if lbls.size == 0:
                print(img_id)
                print("some thing wrong with this sample")
            if lbls[disease_idx] != 0:
                # this images contains at least one bbox
                mask = BBox_mask[:,:,slice_id]
                if np.sum(mask) != 0:
                    # get bounding box
                    mask_idcs = np.where(mask == 1)
                    x_min = np.min(mask_idcs[1])
                    y_min = np.min(mask_idcs[0])
                    x_max = np.max(mask_idcs[1])
                    y_max = np.max(mask_idcs[0])
                    w = x_max - x_min
                    h = y_max - y_min
                    bbox_df = pd.concat([bbox_df, pd.DataFrame([{'Image Index': img_id, 'Finding Label': disease, 'Bbox [x': x_min, 'y': y_min, 'w': w, 'h]': h}])], ignore_index=True)

bbox_df.to_csv(dest_csv, index=False)










# for idx in np.arange(len(source_df)):
#     img_id = source_df.iloc[idx]['image_id']  # print(img_id) # MTR_005_000
#     lbls = source_df.iloc[idx][TRAIN_DISEASES].values.tolist()
#     lbls = np.array(lbls).astype(np.float32)
#     if np.sum(lbls) != 0:
#         # this images contains at least one bbox
#         # get bounding box
#         scan_id = img_id[:-4]
#         current_bbox_dir = os.path.join(bbox_dir, scan_id)
#         all_bbox = [f for f in os.listdir(current_bbox_dir) if f.endswith('.npy')]
#         pos_indices = np.where(lbls == 1)[0]
#         for label_idx in pos_indices:
#             Finding_Label = TRAIN_DISEASES[label_idx]
#             bbox_ids = superIDtoid[label_idx + 1]
#
#
#
#
#             current_bbox_dir = os.path.join(bbox_dir, scan_id)
