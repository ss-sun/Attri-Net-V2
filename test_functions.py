
# import json
#
# data = {
#     "chexpert": [0.2, 0.3, 0.4, 0.12],
#     "chestx-ray": [0.14, 0.22, 0.43]
# }
#
# # Write to file
# with open('/home/susu/Desktop/output.json', 'w') as f:
#     json.dump(data, f, indent=4)
#
# print("Dictionary written to output.json")
#
#
# with open('/home/susu/Desktop/output.json', 'r') as f:
#     data = json.load(f)
#
# print(data)


import numpy as np
from scipy import stats

# Example data
data = np.array([0.8, 0.82, 0.81, 0.83, 0.79])

# Mean and standard error
mean = np.mean(data)
sem = stats.sem(data)  # Standard error of the mean

# 95% confidence interval (two-sided)
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
print(f"Mean: {mean}, 95% CI: {ci}")
print(f"CI_range: {(ci[1] - ci[0])/2}")
print(f"Mean: {mean}, 95% CI: {ci[0]} - {ci[1]}")




# # import numpy as np
# #
# #
# #
# # array_2d = np.array([
# #     [10, 20, 30],
# #     [140, 100, 60],
# #     [70, 80, 90]
# # ])
# #
# # # Find the index of the maximum value
# # max_index = np.argmax(array_2d)
# #
# # # Convert the 1D index to 2D indices
# # max_index_2d = np.unravel_index(max_index, array_2d.shape)
# #
# # print("2D Array:")
# # print(array_2d)
# # print("Position of maximum value:", max_index_2d)
#
# #
# # import numpy as np
# # import torch
# #
# #
# # def to_numpy(tensor):
# #     """
# #     Converting tensor to numpy.
# #     """
# #     if not isinstance(tensor, torch.Tensor):
# #         return tensor
# #     return tensor.detach().cpu().numpy()
# #
# # def reshape_weights(weights_1d):
# #     len = max(weights_1d.shape)
# #     len_sqrt = int(np.sqrt(len))
# #     weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
# #     return weights_2d
# #
# # def upsample_weights(weights_2d, target_size):
# #     print(weights_2d.shape)
# #     source_size = weights_2d.shape[0]
# #     up_ratio = int(target_size/source_size)
# #     upsampled_weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))
# #     return upsampled_weights
# #
# #
# #
# # def get_weightedmap(self, mask, lgs): # lgs is the logistic regression classifier
# #     weights = to_numpy(lgs.linear.weight.data)
# #     weights = reshape_weights(weights)
# #     weights = upsample_weights(weights, target_size=320)
# #     weightedmap = np.multiply(to_numpy(mask.squeeze()), weights)
# #     return weightedmap
# #
# # # below is how I use
# # if self.attr_method == 'attri-net':
# #     task_code = self.solver.latent_z_task[disease].to(self.device)
# #     _, attri = self.solver.net_g(test_data, task_code)
# #     lgs = self.solver.net_lgs[disease] # here you can get the logistic regression classifier from attri-net
# #     attri = self.get_weightedmap(attri, lgs)
# #     hit, pos = self.get_hit(attri, gt_mask, weighted=True)
#
# # import json
# #
# # gt_seg_file = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
# # with open(gt_seg_file) as json_file:
# #     gt_seg_dict = json.load(json_file)
# #
# # print(len(gt_seg_dict.keys()))
# #
# # # 'patient64744_study1_view1_frontal'
# # print(gt_seg_dict['patient64744_study1_view1_frontal']['Cardiomegaly'])
#
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # Create a sample image (replace this with your bwr image data)
# # bwr_image = np.random.rand(100, 100)  # Replace with your bwr image data
# #
# # # Create a sample green mask (replace this with your mask data)
# # green_mask = np.zeros_like(bwr_image)
# # green_mask[40:60, 40:60] = 1  # Example: Set a region to be green
# #
# # # Overlay the green mask on top of the bwr image
# # overlay_image = np.stack([green_mask, bwr_image, bwr_image], axis=-1)
# #
# # # Display the overlay image
# # plt.imshow(overlay_image)
# # plt.show()
#
#
#
#
# # get statics of the chexpert mask dataset.
#
# import json
# from pycocotools import mask
# import numpy as np
#
# gt_seg_file_test = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
# gt_seg_file_valid = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_val.json"
# disease_list = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
#
# with open(gt_seg_file_valid) as json_file:
#     gt_seg_dict = json.load(json_file)
#
# def get_gt_mask(gt_seg_dict, cxr_id, disease):
#     gt_item = gt_seg_dict[cxr_id][disease]
#     gt_mask = mask.decode(gt_item)
#     return gt_mask
#
# print(len(gt_seg_dict.keys())) # 499 unique cxr ids in test set, 187 images in valid set
#
# count = 0
#
# for cxr_id in gt_seg_dict.keys():
#     # print(cxr_id)
#     if "lateral" in cxr_id:
#         continue
#     for disease in disease_list:
#         # print(disease)
#         gt_mask = get_gt_mask(gt_seg_dict, cxr_id, disease)
#         if np.sum(gt_mask) != 0:
#             count += 1
# print("count: ", count)
# # 522 frontal gt masks in total in file: "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
# # 279 frontal gt masks in total in file: "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_val.json"
#
#
#
#

import os
import numpy as np
import pandas as pd

# image_dir= "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled" # contains all images for train, valid and test.
# data_entry_csv_file = "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/Data_Entry_2017.csv"
#
# img_list = os.listdir(image_dir)
# print(len(img_list))
#
# df = pd.read_csv(data_entry_csv_file)
# imgs = df['Image Index'].tolist()
# uni_imgs = list(set(imgs))
# print(len(uni_imgs))

# bbox_file = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_List_2017.csv"
# # BBox_csv_file_train = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_valid_df_scaled.csv"
# # BBox_csv_file_test = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_test_df_scaled.csv"
#
# df = pd.read_csv(bbox_file)
# imgs = df['Image Index'].tolist()
# print(len(imgs))
# uni_imgs = list(set(imgs))
# print(len(uni_imgs))

# diseases = df['Finding Label'].tolist()
# uni_diseases = np.unique(np.array(diseases), return_counts=True)
# print(uni_diseases)


total_csv_file = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train_resized.csv"
remain_csv_file = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/pseudo_guidance/remaining_df.csv"
bbox_train_file = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/pseudo_guidance/pseudo_gudance_df.csv"

df_total = pd.read_csv(total_csv_file)
df_remain = pd.read_csv(remain_csv_file)
df_bbox_train = pd.read_csv(bbox_train_file)

print("len(df_total): ",len(df_total))
print("len(df_remain): ",len(df_remain))
print("len(df_bbox_train): ",len(df_bbox_train))
print(len(df_remain) + len(df_bbox_train))

imgs = df_bbox_train['image_id'].tolist()
print(len(imgs))
uni_imgs = list(set(imgs))
print(len(uni_imgs))


