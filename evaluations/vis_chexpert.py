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
from solvers.bcosnet_solver import bcos_resnet_solver
from train_utils import to_numpy
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from eval_utils import get_weighted_map, draw_BB, draw_hit
from pycocotools import mask

def str2bool(v):
    return v.lower() in ('true')


class vis_analyser():
    def __init__(self, solver, config):
        self.config = config
        self.top_k = config.top_k
        self.dataset = config.dataset # different dataset has different gt annotation format
        self.attr_method = config.attr_method
        self.solver = solver
        self.train_diseases = self.solver.TRAIN_DISEASES
        self.result_dir = os.path.join(config.result_dir, self.attr_method)
        os.makedirs(self.result_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.result_dir, self.attr_method + '_vis_mask')
        os.makedirs(self.plots_dir, exist_ok=True)
        self.draw = False
        if self.dataset == "chexpert":
            gt_seg_file = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
            with open(gt_seg_file) as json_file:
                self.gt_seg_dict = json.load(json_file)

    def run(self):
        self.compute_pixel_sensitivity(attr_method=self.attr_method)

    def create_mask_fromBB(self, img_size, bbox):
        #bbox: [x, y, w, h]
        mask = np.zeros(img_size)
        mask[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
        return mask



    def get_cxr_id(self, path):
        # Remove the "test/" prefix and ".jpg" suffix
        filename = path.replace("test/", "").replace(".jpg", "")
        # Split the filename into its components
        id_components = filename.split("/")
        # Join the components together with underscores
        cxr_id = "_".join(id_components)
        return cxr_id


    def get_gt_mask(self, cxr_id, disease):
        gt_item = self.gt_seg_dict[cxr_id][disease]
        gt_mask = mask.decode(gt_item)
        scaled_mask = self.scale_mask(gt_mask, (320, 320))
        return scaled_mask

    def scale_mask(self, mask, target_size):
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


    def vis_chexpert_sample(self, src_img, attr, gt_annotation, prefix, output_dir):
        gt_mask = gt_annotation
        src_img = to_numpy(src_img * 0.5 + 0.5).squeeze()
        src_img = Image.fromarray(src_img * 255).convert('RGB')
        rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = gt_mask * 255
        mask_img = Image.fromarray(rgb_mask).convert('RGB')
        mask_img.putalpha(50)
        src_img.paste(mask_img, (0, 0), mask_img)
        src_img.save(os.path.join(output_dir, prefix + '_src.jpg'))
        attr = to_numpy(-attr * 0.5 + 0.5).squeeze()
        attri_img = plt.cm.bwr(attr)
        attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
        attri_img.paste(mask_img, (0, 0), mask_img)
        attri_img.save(os.path.join(output_dir, prefix + '_attri.jpg'))




    def plot_hit(self, attri, attri_mask, gt_mask, src_img, pos, prefix):
        src_img = to_numpy(src_img * 0.5 + 0.5).squeeze()
        src_img = Image.fromarray(src_img * 255).convert('RGB')
        rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = gt_mask * 255
        rgb_mask[:, :, 2] = attri_mask * 255
        # rgb_mask[pos[0], pos[1], 0] = 255
        img = Image.fromarray(rgb_mask).convert('RGB')
        # Create a new ImageDraw object
        draw = ImageDraw.Draw(img)
        # Set the size of the dot
        size = 5
        # Calculate the coordinates of the circle
        x0 = pos[1] - size
        y0 = pos[0] - size
        x1 = pos[1] + size
        y1 = pos[0] + size
        img.putalpha(50)
        # Draw the circle
        draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
        src_img.paste(img, (0, 0), img)
        # src_img.show()
        src_img.save(os.path.join(self.attr_dir, prefix +'.jpg'))

        flip_mask = -0.5 * to_numpy(attri).squeeze() + 0.5
        flip_mask = Image.fromarray(flip_mask * 255).convert('RGB')
        draw = ImageDraw.Draw(flip_mask)
        draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
        # flip_mask.show()
        flip_mask.save(os.path.join(self.attr_dir, prefix + '_mask.jpg'))





    def vis_chexpert(self, attr_method):

        samples_per_disease = {}
        hits_per_disease = {}
        pixel_localization_accuracy = {}
        total = 0
        hits = 0

        for disease in self.solver.TRAIN_DISEASES:
            hits_per_disease[disease] = 0
            samples_per_disease[disease] = 0
            pixel_localization_accuracy[disease] = 0

        for index, row in self.solver.dataloaders["BB_test"].dataset.df.iterrows():
            if "lateral" in row['Path']:
                continue
            path = row['Path']
            cxr_id = self.get_cxr_id(path)
            if cxr_id in self.gt_seg_dict.keys():
                # this image has gt segmentation, we will compute hit.
                data = self.solver.dataloaders["BB_test"].dataset[int(index)]
                img, label = data["img"], data["label"]
                img = torch.from_numpy(img[None])
                for disease in self.solver.TRAIN_DISEASES:
                    if disease in self.gt_seg_dict[cxr_id].keys():
                        gt_mask = self.get_gt_mask(cxr_id, disease)
                        if np.sum(gt_mask) != 0:
                            samples_per_disease[disease] += 1
                            total += 1
                            label_idx = self.solver.TRAIN_DISEASES.index(disease)
                            attr_raw = self.solver.get_attributes(img, label_idx)
                            attr_raw = to_numpy(attr_raw).squeeze()
                            prefix = cxr_id +'_' + attr_method + '_' + disease
                            self.vis_chexpert_sample(img, attr_raw, gt_annotation=gt_mask, prefix=prefix, output_dir=self.plots_dir)















    def compute_pixel_sensitivity(self, attr_method):

        # if self.dataset == "nih_chestxray" or self.dataset == "vindr_cxr" or self.dataset == "skmtea":
        #
        #     self.compute_hit_nih_vindr_skmtea(attr_method)
        #     self.compute_EPG_nih_vindr_skmtea(attr_method)
        #
        # # if self.dataset == "vindr":
        # #     self.compute_hit_vindr(attr_method)
        #
        #
        # if self.dataset == "skmtea":
        #     self.compute_hit_skmtae(attr_method)

        if self.dataset == "chexpert":
            # self.compute_hit_chexpert(attr_method)
            self.vis_chexpert(attr_method)






























def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='attri-net', choices=['resnet_cls', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--attr_method', type=str, default='attri-net',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'attri-net' ,'gifsplanation', 'bcos'")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea'])
    parser.add_argument('--top_k', type=int, default=1, help="top k pixels to be considered as hit")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    parser.add_argument('--use_wandb', type=str2bool, default=False, help='whether to use wandb')
    return parser


def get_arguments():
    #from model_dict import resnet_models, attrinet_models
    from model_dict import resnet_model_path_dict, attrinet_model_path_dict, bcos_resnet_model_path_dict
    parser = argument_parser()
    exp_configs = parser.parse_args()

    if exp_configs.exp_name == 'resnet_cls':
        exp_configs.model_path = resnet_model_path_dict[exp_configs.dataset]
        exp_configs.result_dir = os.path.join(exp_configs.model_path, "pixel_sensitivity_result_dir")

    if exp_configs.exp_name == 'bcos_resnet':
        exp_configs.model_path = bcos_resnet_model_path_dict[exp_configs.dataset]
        exp_configs.result_dir = os.path.join(exp_configs.model_path, "pixel_sensitivity_result_dir")

    if exp_configs.exp_name == 'attri-net':
        print("evaluating our model")
        exp_configs.model_path = attrinet_model_path_dict[exp_configs.dataset]
        print("evaluate model: " + exp_configs.model_path)

        exp_configs.result_dir = os.path.join(exp_configs.model_path, "pixel_sensitivity_result_dir")
        # configurations of generator
        exp_configs.image_size = 320
        exp_configs.generator_type = 'stargan'
        exp_configs.deep_supervise = False

        # configurations of latent code generator
        exp_configs.n_fc = 8
        exp_configs.n_ones = 20
        exp_configs.num_out_channels = 1

        # configurations of classifiers
        exp_configs.lgs_downsample_ratio = 32

    return exp_configs


def prep_solver(datamodule, exp_configs):
    data_loader = {}
    if "resnet" in exp_configs.exp_name:

        # data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        # data_loader['valid'] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        # data_loader['test'] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)

        data_loader["train"] = None
        data_loader['valid'] = None
        data_loader['test'] = None

        if exp_configs.dataset == "nih_chestxray" or exp_configs.dataset == "vindr_cxr" or exp_configs.dataset == "chexpert" or exp_configs.dataset == "skmtea":
            data_loader["BB_test"] = datamodule.BBox_test_dataloader(batch_size=1, shuffle=False)
        if "bcos" in exp_configs.exp_name:
            solver = bcos_resnet_solver(exp_configs, data_loader=data_loader)
        else:
            solver = resnet_solver(exp_configs, data_loader=data_loader)
            solver.set_explainer(which_explainer=exp_configs.attr_method)

    if "attri-net" in exp_configs.exp_name:

        # val_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        # test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader['train_pos'] = None
        data_loader['train_neg'] = None
        data_loader['vis_pos'] = None
        data_loader['vis_neg'] = None
        data_loader['valid'] = None
        data_loader['test'] = None
        if exp_configs.dataset == "nih_chestxray" or exp_configs.dataset == "vindr_cxr" or exp_configs.dataset == "chexpert" or exp_configs.dataset == "skmtea":
            data_loader["BB_test"] = datamodule.BBox_test_dataloader(batch_size=1, shuffle=False)
        solver = task_switch_solver(exp_configs, data_loader=data_loader)

    return solver



def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    solver = prep_solver(datamodule, config)
    analyser = vis_analyser(solver, config)
    analyser.run()


if __name__ == "__main__":

    params = get_arguments()
    main(params)