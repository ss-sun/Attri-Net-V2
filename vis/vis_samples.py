import os
import sys
sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/github_projects/tmi"))
import json
import numpy as np
import torch
import random
import cv2
import argparse
from train_utils import prepare_datamodule, to_numpy
from solvers.resnet_solver import resnet_solver
# from solvers.attrinet_solver import task_switch_solver
# from solvers.attrinet_solver_energyloss import task_switch_solver
from solvers.attrinet_solver import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from evaluations.eval_utils import get_weighted_map, draw_BB, draw_hit, vis_samples_withMask
from pycocotools import mask
import datetime

def str2bool(v):
    return v.lower() in ('true')

class visualizer():
    def __init__(self, solver, config):
        self.config = config
        self.dataset = config.dataset # different dataset has different gt annotation format
        self.solver = solver
        self.net_lgs = solver.net_lgs
        self.net_g = solver.net_g
        self.latent_z_task = solver.latent_z_task
        self.attr_method = config.attr_method
        self.TRAIN_DISEASES = self.solver.TRAIN_DISEASES
        self.plots_dir = os.path.join(config.model_path, "vis_samples_in_testset", self.attr_method)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.best_threshold = {}
        threshold_path = os.path.join(config.model_path, "class_sensitivity_result_dir", "best_threshold.txt")
        if os.path.exists(threshold_path):
            print("threshold alreay computed, load threshold")
            # threshold = np.loadtxt(open(threshold_path)).reshape(1,)
            threshold = np.loadtxt(open(threshold_path))
            for c in range(len(self.solver.TRAIN_DISEASES)):
                disease = self.solver.TRAIN_DISEASES[c]
                self.best_threshold[disease] = threshold[c]

    def get_img_id(self, idx):
        df = self.solver.test_loader.dataset.df
        if self.dataset == "nih_chestxray":
            img_id = df.iloc[idx]['Image Index']
        if self.dataset == "vindr_cxr":
            img_id = self.solver.test_loader.dataset.image_list[idx] # one unique image id may have multiple bounding boxes corresponding to different entries in the dataframe
        if self.dataset == "chexpert":
            img_path = df.iloc[idx]['Path'].split('/')
            img_id = img_path[1]+'_' + img_path[2] + '_' + img_path[3][:-4]
        return img_id

    def vis_color_map(self, map, output_dir, prefix):
        attr = to_numpy(map * 0.5 + 0.5).squeeze()
        attri_img = plt.cm.bwr(
            attr)  # use bwr color map, here negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
        attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
        # attri_img.show()
        attri_img.save(os.path.join(output_dir, prefix + '.jpg'))


    def run(self):

        for idx, row in self.solver.dataloaders["test"].dataset.df.iterrows():
            img_id = self.get_img_id(idx)
            data = self.solver.dataloaders["test"].dataset[int(idx)]
            test_data, test_labels = data["img"], data["label"]
            test_data = torch.from_numpy(test_data)
            raw_attri_maps = []
            prefixes = []

            for disease in self.TRAIN_DISEASES:
                threshold = self.best_threshold[disease]
                gt_label= test_labels[self.TRAIN_DISEASES.index(disease)]
                dests, attrs = self.solver.get_attributes(test_data, label_idx=self.TRAIN_DISEASES.index(disease))
                prob = self.solver.get_probs(test_data, label_idx=self.TRAIN_DISEASES.index(disease))

                if prob > threshold:
                    y_pred = 1
                else:
                    y_pred = 0
                prefix = img_id+ "_"+ disease + "_" + "y_pred_" + str(y_pred) + "_GT: " + str(gt_label) + "y_pred_prob: " + str(prob.item()) + "_threshold: " + str(threshold)
                raw_attri_maps.append(to_numpy(-attrs.squeeze()))
                prefixes.append(prefix)

            max_abs_value = np.abs(np.array(raw_attri_maps)).max()
            for i in range(len(raw_attri_maps)):
                map = raw_attri_maps[i]
                scaled_attri_map = (map / max_abs_value)
                self.vis_color_map(scaled_attri_map, self.plots_dir, prefixes[i])





















def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="pixel sensitivitiy metric analyser.")
    parser.add_argument('--exp_name', type=str, default='attri-net', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--attr_method', type=str, default='attri-net',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'attri-net' ,'gifsplanation', 'bcos'")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--img_mode', type=str, default='gray',
                        choices=['color', 'gray'])  # will change to color if dataset is airogs_color
    # Data configuration.
    parser.add_argument('--dataset', type=str, default='chexpert',
                        choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'contaminated_chexpert'])
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=False, help='whether to run on the GPU')
    parser.add_argument('--use_wandb', type=str2bool, default=False, help='whether to use wandb')
    return parser



def update_attrinet_params(opts):

    # Configurations of networks
    opts.image_size = 320
    opts.n_fc = 8
    opts.n_ones = 20
    opts.num_out_channels = 1
    opts.lgs_downsample_ratio = 32
    return opts




def prep_solver(datamodule, exp_configs):

    data_loader = {}
    if "resnet" in exp_configs.exp_name:
        data_loader["train"] = None
        data_loader['valid'] = None
        data_loader["test"] = datamodule.test_dataloader(batch_size=1, shuffle=False)
        if "bcos" in exp_configs.exp_name:
            solver = bcos_resnet_solver(exp_configs, data_loader=data_loader)
        else:
            solver = resnet_solver(exp_configs, data_loader=data_loader)
            solver.set_explainer(which_explainer=exp_configs.attr_method)

    if "attri-net" in exp_configs.exp_name:
        test_loader = datamodule.test_dataloader(batch_size=1, shuffle=False)
        data_loader['train_pos'] = None
        data_loader['train_neg'] = None
        data_loader['vis_pos'] = None
        data_loader['vis_neg'] = None
        data_loader['valid'] = None
        data_loader['test'] = test_loader
        solver = task_switch_solver(exp_configs, data_loader=data_loader)
    return solver

def update_params_with_model_path(opts, model_path):
    if 'attri-net' in model_path:
        opts.exp_name = 'attri-net'
        if opts.attr_method != 'attri-net':
            opts.attr_method = 'attri-net'
        opts = update_attrinet_params(opts)
    if 'resnet' in model_path and 'bcos' not in model_path:
        opts.exp_name = 'resnet'
        assert opts.attr_method in ['lime', 'GCam', 'GB', 'shap', 'gifsplanation']
    if 'bcos' in model_path:
        opts.exp_name = 'bcos_resnet'
        if opts.attr_method != 'bcos_resnet':
            opts.attr_method = 'bcos_resnet'
    if 'chexpert' in model_path:
        opts.dataset = 'chexpert'
    if 'nih' in model_path:
        opts.dataset = 'nih_chestxray'
    if 'vindr' in model_path:
        opts.dataset = 'vindr_cxr'

    print("evaluating model: " + opts.exp_name + " on dataset: " + opts.dataset)

    return opts




def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    solver = prep_solver(datamodule, config)
    vis = visualizer(solver, config)
    vis.run()

    return results


if __name__ == "__main__":

    model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    parser = argument_parser()
    opts = parser.parse_args()
    print("Now evaluating model: " + model_path)
    opts.model_path = model_path
    opts = update_params_with_model_path(opts, model_path)
    results = main(opts)


