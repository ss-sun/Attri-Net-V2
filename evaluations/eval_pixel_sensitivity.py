import os
import json
import numpy as np
import torch
import random
import cv2
import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from train_utils import to_numpy
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from eval_utils import get_weighted_map, draw_BB

def str2bool(v):
    return v.lower() in ('true')


class pixel_sensitivity_analyser():
    def __init__(self, solver, config):
        self.config = config
        self.dataset = config.dataset # different dataset has different gt annotation format
        self.attr_method = config.attr_method
        self.solver = solver
        self.result_dir = config.result_dir
        self.plots_dir = os.path.join(self.result_dir, self.attr_method + '_BB_plots')
        self.train_diseases = self.solver.TRAIN_DISEASES
        # self.best_threshold = {}
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.draw_gt = True

        # # Read data from file:
        # threshold_path = os.path.join(self.result_dir, "best_threshold.txt")
        # if os.path.exists(threshold_path):
        #     print("threshold alreay computed, load threshold")
        #     threshold = np.loadtxt(open(threshold_path))
        #     for c in range(len(self.solver.TRAIN_DISEASES)):
        #         disease = self.solver.TRAIN_DISEASES[c]
        #         self.best_threshold[disease] = threshold[c]
        # else:
        #     self.best_threshold = solver.get_optimal_thresholds(save_result=True, result_dir=self.result_dir)






    def draw_hit(self, pos, attr_map, gt_annotation, prefix, output_dir):
        attr_map = to_numpy(attr_map * 0.5 + 0.5).squeeze()
        attr_map = Image.fromarray(attr_map * 255).convert('RGB')

        # Set the size of the red dot
        size = 5
        # Calculate the coordinates of the circle
        x0 = pos[1] - size
        y0 = pos[0] - size
        x1 = pos[1] + size
        y1 = pos[0] + size

        if self.dataset == "nih_chestxray":
            bbox = gt_annotation
            x_min = int(bbox[0].item())
            y_min = int(bbox[1].item())
            x_max = x_min + int(bbox[2].item())
            y_max = y_min + int(bbox[3].item())

            draw = ImageDraw.Draw(attr_map)
            draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
            draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
            attr_map.save(os.path.join(output_dir, prefix + '.png'))



        if self.dataset == "chexpert":
            gt_mask = gt_annotation
            rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            rgb_mask[:, :, 1] = gt_mask * 255
            rgb_mask[:, :, 2] = attr_map * 255
            img = Image.fromarray(rgb_mask).convert('RGB')
            # Create a new ImageDraw object
            draw = ImageDraw.Draw(img)
            img.putalpha(50)
            # Draw the circle
            draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
            src_img.paste(img, (0, 0), img)
            # src_img.show()
            src_img.save(os.path.join(output_dir, prefix + '.jpg'))



    def compute_pixel_sensitivity(self, attr_method):
        samples_per_disease = {}
        hits_per_disease = {}
        pixel_localization_accuracy = {}
        for disease in self.solver.TRAIN_DISEASES:
            hits_per_disease[disease] = 0
            samples_per_disease[disease] = 0
            pixel_localization_accuracy[disease] = 0
        total = 0
        hits = 0

        if self.dataset == "chexpert":
            for idx, row in self.solver.test_loader.dataset.df.iterrows():
                data = self.solver.dataloaders["BB_test"].dataset[int(idx)]
                img = data["img"]
                test_data = torch.from_numpy(test_data[None])
                test_labels = torch.from_numpy(test_labels[None])
                test_data = test_data.to(self.device)



        if self.dataset == "nih_chestxray":
            for idx, row in self.solver.dataloaders["BB_test"].dataset.df.iterrows():
                disease = row['Finding Label']
                img_id = row['Image Index'][:-4]
                if disease not in self.solver.TRAIN_DISEASES:
                    continue # means no self.solver.TRAIN_DISEASES in this image
                else:
                    total += 1
                    samples_per_disease[disease] += 1
                    label_idx = self.solver.TRAIN_DISEASES.index(disease)
                    data = self.solver.dataloaders["BB_test"].dataset[int(idx)]
                    img = data["img"]
                    bbox = data["BBox"]
                    img = torch.from_numpy(img[None])
                    attr_raw = self.solver.get_attributes(img, label_idx)
                    attr_raw = to_numpy(attr_raw).squeeze()
                    # print("np.max(attr_raw)", np.max(attr_raw))
                    # print("np.min(attr_raw)", np.min(attr_raw))
                    if self.draw_gt:
                        draw_BB(to_numpy(img.squeeze() * 0.5 + 0.5), bbox, img_id, '_input: ' + disease, self.plots_dir)
                    if attr_method == 'ours':
                        attr = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                    else:
                        attr_max = np.max(attr_raw)
                        attr_min = np.min(attr_raw)
                        if attr_max!= attr_min:
                            attr = (attr_raw - attr_min) / (attr_max - attr_min)
                        else:
                            attr = attr_raw
                    # hit, pos = self.get_hit_top100(attr_map=attr, gt_annotation=bbox)
                    hit, pos = self.get_hit(attr_map=attr, gt_annotation=bbox)
                    hits += hit
                    hits_per_disease[disease] += hit
                    prefix = img_id +  '_hit_' + str(hit) + '_' + attr_method + '_' + disease
                    self.draw_hit(pos, attr, bbox, prefix, self.plots_dir)

        print("hits", hits)
        print("total", total)
        print("hit ratio", hits/total)
        for disease in self.solver.TRAIN_DISEASES:
            if samples_per_disease[disease] != 0:
                pixel_localization_accuracy[disease] = hits_per_disease[disease]/samples_per_disease[disease]
        print("pixel_localization_accuracy", pixel_localization_accuracy)
        return pixel_localization_accuracy

    def get_hit_top100(self, attr_map, gt_annotation):
        pixel_importance = attr_map

        idcs = np.argsort(pixel_importance.flatten())  # from smallest to biggest
        idcs = idcs[::-1]  # if we want the order biggest to smallest, we reverse the indices array
        idcs = idcs[:100]

        # Compute the corresponding masks for deleting pixels in the given order
        positions = np.array(
            np.unravel_index(idcs, pixel_importance.shape)).T  # first colum, h index, second column, w index
        attri_pos_h = positions[:, 0]
        attri_pos_w = positions[:, 1]
        top_attri_pixels = list(zip(attri_pos_h, attri_pos_w))
        top_attri_set = set(top_attri_pixels)

        if self.dataset == "chexpert":
            # gt_annotation is segmantation mask
            gt_mask = gt_annotation
            assert (gt_mask.shape == pixel_importance.shape)
            if np.sum(gt_mask) == 0:
                hit = np.nan
                pos = np.nan
            else:
                pos = (x, y)
                if (gt_mask[x][y] == 1):
                    hit = 1
                else:
                    hit = 0
        if self.dataset == "nih_chestxray":
            # gt_annotation is bounding box
            BBox = gt_annotation
            #(x_min, y_min) the upper left corner of the bounding box
            x_min = BBox[0].item()
            y_min = BBox[1].item()
            x_max = x_min + BBox[2].item()
            y_max = y_min + BBox[3].item()

            gt_mask = np.zeros(pixel_importance.shape)
            gt_mask[y_min:y_max, x_min:x_max] = 1 # swap from (height, width) to (row, column).

            gt_mask_pos = np.where(gt_mask == 1)
            gt_mask_pos_x = gt_mask_pos[0]
            gt_mask_pos_y = gt_mask_pos[1]

            gt_pixels = list(zip(gt_mask_pos_x, gt_mask_pos_y))
            gt_set = set(gt_pixels)
            inter_set = gt_set.intersection(top_attri_set)

            if (len(inter_set) > 0):
                hit = 1
                pos = list(inter_set)[0]
            else:
                hit = 0
                pos = top_attri_pixels[0]

        return hit, pos




    def get_hit(self, attr_map, gt_annotation):
        pixel_importance = attr_map
        # get most significant pixel, swap from (row, column) to (height, width)
        h = np.unravel_index(np.argmax(pixel_importance, axis=None), pixel_importance.shape)[0]
        w = np.unravel_index(np.argmax(pixel_importance, axis=None), pixel_importance.shape)[1]

        if self.dataset == "chexpert":
            # gt_annotation is segmantation mask
            gt_mask = gt_annotation
            assert (gt_mask.shape == pixel_importance.shape)
            if np.sum(gt_mask) == 0:
                hit = np.nan
                pos = np.nan
            else:
                pos = (x, y)
                if (gt_mask[x][y] == 1):
                    hit = 1
                else:
                    hit = 0
        else:
            # gt_annotation is bounding box
            BBox = gt_annotation
            #(x_min, y_min) the upper left corner of the bounding box
            x_min = BBox[0].item()
            y_min = BBox[1].item()
            x_max = x_min + BBox[2].item()
            y_max = y_min + BBox[3].item()

            if w >= x_min and w <= x_max and h >= y_min and h <= y_max:
                hit = 1
                pos = (h,w)
            else:
                hit = 0
                pos = (h,w)

        return hit, pos



    def run(self):
        self.compute_pixel_sensitivity(attr_method=self.attr_method)








def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='resnet_cls', choices=['resnet_cls', 'ours', 'codanet'])
    parser.add_argument('--attr_method', type=str, default='shap',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'ours' , 'gifsplanation'")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='nih_chestxray', choices=['chexpert', 'nih_chestxray', 'vindr_cxr'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    return parser


def get_arguments():
    from model_dict import resnet_models, attrinet_models
    parser = argument_parser()
    opts = parser.parse_args()

    if opts.exp_name == 'resnet_cls':
        opts.model_path = resnet_models[opts.dataset]
        opts.result_dir = os.path.join(opts.model_path, "PointingGame_result_dir")

    if opts.exp_name == 'ours':
        print("evaluating our model")
        opts.model_path = attrinet_models[opts.dataset]
        print("evaluate model: " + opts.model_path)

        opts.result_dir = os.path.join(opts.model_path, "PointingGame_result_dir")
        # configurations of generator
        opts.image_size = 320
        opts.generator_type = 'stargan'
        opts.deep_supervise = False

        # configurations of latent code generator
        opts.n_fc = 8
        opts.n_ones = 20
        opts.num_out_channels = 1

        # configurations of classifiers
        opts.lgs_downsample_ratio = 32

    return opts


def prep_solver(datamodule, exp_configs):
    data_loader = {}
    if "resnet" in exp_configs.exp_name:

        data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        data_loader['valid'] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader['test'] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        if exp_configs.dataset == "nih_chestxray" or exp_configs.dataset == "vindr_cxr":
            data_loader["BB_test"] = datamodule.BBox_test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        solver = resnet_solver(exp_configs, data_loader=data_loader)
        solver.set_explainer(which_explainer=exp_configs.attr_method)

    if "ours" in exp_configs.exp_name:

        val_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader['train_pos'] = None
        data_loader['train_neg'] = None
        data_loader['vis_pos'] = None
        data_loader['vis_neg'] = None

        data_loader['valid'] = val_loader
        data_loader['test'] = test_loader
        if exp_configs.dataset == "nih_chestxray" or exp_configs.dataset == "vindr_cxr":
            data_loader["BB_test"] = datamodule.BBox_test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        solver = task_switch_solver(exp_configs, data_loader=data_loader)

    return solver



def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    solver = prep_solver(datamodule, config)
    analyser = pixel_sensitivity_analyser(solver, config)
    analyser.run()


if __name__ == "__main__":

    params = get_arguments()
    main(params)