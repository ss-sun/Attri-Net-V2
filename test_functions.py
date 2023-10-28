# import numpy as np
#
#
#
# array_2d = np.array([
#     [10, 20, 30],
#     [140, 100, 60],
#     [70, 80, 90]
# ])
#
# # Find the index of the maximum value
# max_index = np.argmax(array_2d)
#
# # Convert the 1D index to 2D indices
# max_index_2d = np.unravel_index(max_index, array_2d.shape)
#
# print("2D Array:")
# print(array_2d)
# print("Position of maximum value:", max_index_2d)

#
# import numpy as np
# import torch
#
#
# def to_numpy(tensor):
#     """
#     Converting tensor to numpy.
#     """
#     if not isinstance(tensor, torch.Tensor):
#         return tensor
#     return tensor.detach().cpu().numpy()
#
# def reshape_weights(weights_1d):
#     len = max(weights_1d.shape)
#     len_sqrt = int(np.sqrt(len))
#     weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
#     return weights_2d
#
# def upsample_weights(weights_2d, target_size):
#     print(weights_2d.shape)
#     source_size = weights_2d.shape[0]
#     up_ratio = int(target_size/source_size)
#     upsampled_weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))
#     return upsampled_weights
#
#
#
# def get_weightedmap(self, mask, lgs): # lgs is the logistic regression classifier
#     weights = to_numpy(lgs.linear.weight.data)
#     weights = reshape_weights(weights)
#     weights = upsample_weights(weights, target_size=320)
#     weightedmap = np.multiply(to_numpy(mask.squeeze()), weights)
#     return weightedmap
#
# # below is how I use
# if self.attr_method == 'attri-net':
#     task_code = self.solver.latent_z_task[disease].to(self.device)
#     _, attri = self.solver.net_g(test_data, task_code)
#     lgs = self.solver.net_lgs[disease] # here you can get the logistic regression classifier from attri-net
#     attri = self.get_weightedmap(attri, lgs)
#     hit, pos = self.get_hit(attri, gt_mask, weighted=True)

# import json
#
# gt_seg_file = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
# with open(gt_seg_file) as json_file:
#     gt_seg_dict = json.load(json_file)
#
# print(len(gt_seg_dict.keys()))
#
# # 'patient64744_study1_view1_frontal'
# print(gt_seg_dict['patient64744_study1_view1_frontal']['Cardiomegaly'])


import numpy as np
import matplotlib.pyplot as plt

# Create a sample image (replace this with your bwr image data)
bwr_image = np.random.rand(100, 100)  # Replace with your bwr image data

# Create a sample green mask (replace this with your mask data)
green_mask = np.zeros_like(bwr_image)
green_mask[40:60, 40:60] = 1  # Example: Set a region to be green

# Overlay the green mask on top of the bwr image
overlay_image = np.stack([green_mask, bwr_image, bwr_image], axis=-1)

# Display the overlay image
plt.imshow(overlay_image)
plt.show()