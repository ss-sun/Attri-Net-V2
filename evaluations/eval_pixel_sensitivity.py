import os
import sys
# sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/github_projects/tmi"))
sys.path.append(os.path.abspath("/mnt/lustre/work/baumgartner/sun22/project/github_projects/AttriNet_revision/Attri-Net-V2"))

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
from eval_utils import get_weighted_map, draw_BB, draw_hit, vis_samples_withMask
from pycocotools import mask
from model_dict import resnet_models, bcos_resnet_models, attrinet_models, aba_loss_attrinet_models, aba_guidance_attrinet_models, guided_attrinet_models,guided_bcos_resnet_models
import datetime
from scipy import stats

def str2bool(v):
    return v.lower() in ('true')

class pixel_sensitivity_analyser():
    def __init__(self, solver, config):
        self.config = config
        self.dataset = config.dataset # different dataset has different gt annotation format
        self.attr_method = config.attr_method
        self.solver = solver
        self.train_diseases = self.solver.TRAIN_DISEASES
        self.result_dir = os.path.join(config.model_path, "pixel_sensitivity_result_dir", self.attr_method)
        os.makedirs(self.result_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.result_dir, self.attr_method + '_pixel_sensitivity_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        self.draw = False
        if self.dataset == "chexpert":
            # gt_seg_file = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
            gt_seg_file = "/mnt/lustre/work/baumgartner/sun22/data/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
            with open(gt_seg_file) as json_file:
                self.gt_seg_dict = json.load(json_file)

    def run(self):
        results = self.compute_pixel_sensitivity(attr_method=self.attr_method)
        return results

    def compute_pixel_sensitivity(self, attr_method):

        if self.dataset == "nih_chestxray" or "vindr_cxr" in self.dataset:
            avg_EPG_score = self.compute_EPG_nih_vindr(attr_method)
        if self.dataset == "chexpert":
            avg_EPG_score = self.compute_EPG_chexpert(attr_method)

        return avg_EPG_score



    def compute_EPG_nih_vindr(self, attr_method):

        print("Now computing EPG score for: ", self.dataset)
        print("current attr_method: ", attr_method)

        EPG_score = {}
        avg_EPG_score = {}
        avg_EPG_CI = {}
        all_scores = []
        for disease in self.solver.TRAIN_DISEASES:
            EPG_score[disease] = []
            avg_EPG_score[disease] = 0

        for idx, row in self.solver.dataloaders["BB_test"].dataset.df.iterrows():
                if self.dataset == 'nih_chestxray':
                    disease = row['Finding Label']
                    img_id = row['Image Index'][:-4]
                if 'vindr_cxr' in self.dataset:
                    disease = row['class_name']
                    img_id = row['image_id']

                if disease not in self.solver.TRAIN_DISEASES:
                    continue  # means no self.solver.TRAIN_DISEASES in this image
                else:

                    label_idx = self.solver.TRAIN_DISEASES.index(disease)
                    data = self.solver.dataloaders["BB_test"].dataset[int(idx)]
                    img = data["img"]
                    bbox = data["BBox"]
                    gt_mask = self.create_mask_fromBB(img_size=img.squeeze().shape, bbox=bbox)

                    img = torch.from_numpy(img[None])

                    if attr_method in ['attri-net']:
                        dests, attr_raw = self.solver.get_attributes(img, label_idx)
                        attr_raw = to_numpy(attr_raw).squeeze()
                        attr_weighted = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                        attr = np.where(attr_weighted > 0, attr_weighted, 0)
                    else:
                        attr_raw = self.solver.get_attributes(img, label_idx)
                        attr = to_numpy(attr_raw).squeeze()
                        dests = None

                    score = self.get_EPG_score(attr, gt_mask)
                    EPG_score[disease].append(score)
                    # all_scores.append(score)

                    if self.draw:
                        prefix = img_id + '_' + attr_method + '_' + disease
                        vis_samples_withMask(src_img=img, attr=attr, dests=dests, gt_annotation=gt_mask, prefix=prefix, output_dir= self.plots_dir, attr_method=self.attr_method)



        avg_scores = []
        for disease in self.solver.TRAIN_DISEASES:
            scores = EPG_score[disease]
            if scores != []:
                mean_score = np.mean(np.array(scores))
                std_score = np.std(np.array(scores))
                ci = stats.t.interval(0.95, len(scores) - 1, loc=mean_score,
                                      scale=std_score)
                avg_EPG_score[disease] = str(mean_score)
                avg_EPG_CI[disease] = str((ci[1] - ci[0]) / 2)
                avg_scores.append(mean_score)
                all_scores.append(scores)

        print("avg_EPG_score: ", avg_EPG_score)
        print("avg_EPG_CI: ", avg_EPG_CI)
        print("avg_EPG accross disease: ", np.mean(np.array(avg_scores)))

        all_scores = np.concatenate(all_scores)
        all_scores = all_scores.flatten()
        all_scores_mean = np.mean(all_scores)
        all_scores_std = np.std(all_scores)
        ci = stats.t.interval(0.95, len(all_scores) - 1, loc=all_scores_mean,
                              scale=all_scores_std)
        print(f"all_scores_mean: {all_scores_mean}, 95% CI: {ci}")
        print(f"CI_range: {(ci[1] - ci[0]) / 2}")

        result_file_path = os.path.join(self.result_dir, "avg_EPG_results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump(avg_EPG_score, json_file, indent=4)
        return avg_EPG_score



    def compute_EPG_chexpert(self, attr_method):
        print("Now computing EPG score for CheXpert dataset.")
        print("current attr_method: ", attr_method)

        EPG_score = {}
        avg_EPG_score = {}
        avg_EPG_CI = {}
        all_scores = []
        for disease in self.solver.TRAIN_DISEASES:
            EPG_score[disease] = []
            avg_EPG_score[disease] = 0

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
                            # samples_per_disease[disease] += 1
                            # total += 1
                            label_idx = self.solver.TRAIN_DISEASES.index(disease)

                            if attr_method in ['attri-net']:
                                dests, attr_raw = self.solver.get_attributes(img, label_idx)
                                attr_raw = to_numpy(attr_raw).squeeze()
                                attr_weighted = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                                attr = np.where(attr_weighted > 0, attr_weighted, 0)
                            else:
                                attr_raw = self.solver.get_attributes(img, label_idx)
                                attr = to_numpy(attr_raw).squeeze()
                                dests = None

                            score = self.get_EPG_score(attr, gt_mask)
                            EPG_score[disease].append(score)

                            if self.draw:
                                prefix = cxr_id + '_' + attr_method + '_' + disease
                                vis_samples_withMask(src_img=img, attr=attr, dests=dests, gt_annotation=gt_mask,
                                                   prefix=prefix, output_dir=self.plots_dir,
                                                   attr_method=self.attr_method)

        avg_scores = []
        for disease in self.solver.TRAIN_DISEASES:
            scores = EPG_score[disease]
            if scores != []:
                mean_score = np.mean(np.array(scores))
                std_score = np.std(np.array(scores))
                ci = stats.t.interval(0.95, len(scores) - 1, loc=mean_score, scale=std_score)
                avg_EPG_score[disease] = str(mean_score)
                avg_EPG_CI[disease] = str((ci[1] - ci[0]) / 2)
                avg_scores.append(mean_score)
                all_scores.append(scores)

        print("avg_EPG_score: ", avg_EPG_score)
        print("avg_EPG_CI: ", avg_EPG_CI)
        print("avg_EPG accross disease: ", np.mean(np.array(avg_scores)))

        all_scores = np.concatenate(all_scores)
        all_scores = all_scores.flatten()
        all_scores_mean = np.mean(all_scores)
        all_scores_std = np.std(all_scores)
        ci = stats.t.interval(0.95, len(all_scores) - 1, loc=all_scores_mean,
                              scale=all_scores_std)
        print(f"all_scores_mean: {all_scores_mean}, 95% CI: {ci}")
        print(f"CI_range: {(ci[1] - ci[0]) / 2}")

        result_file_path = os.path.join(self.result_dir, "avg_EPG_results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump(avg_EPG_score, json_file, indent=4)
        return avg_EPG_score




    def get_EPG_score(self, attr, gt_annotation):
        # here we only consider the positive contribution. since attribution maps from attri-net are already weighted, therefore, positive value means positive contribution.
        #
        # scale to [0,1]
        attr_max = np.max(attr)
        attr_min = np.min(attr)
        if attr_max != attr_min:
            pixel_importance = (attr - attr_min) / (attr_max - attr_min)
        else:
            pixel_importance = attr

        sum = np.sum(pixel_importance)
        masked_attr_map = np.multiply(pixel_importance, gt_annotation)
        sum_masked = np.sum(masked_attr_map)
        if sum != 0:
            return sum_masked / sum
        else:
            return 0



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




def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="pixel sensitivitiy metric analyser.")
    parser.add_argument('--exp_name', type=str, default='bcos_resnet', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--attr_method', type=str, default='bcos',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'attri-net' ,'gifsplanation', 'bcos'")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--img_mode', type=str, default='gray',
                        choices=['color', 'gray'])  # will change to color if dataset is airogs_color
    # Data configuration.
    parser.add_argument('--dataset', type=str, default='chexpert',
                        choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'contaminated_chexpert'])
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
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
    if exp_configs.dataset == "nih_chestxray" or "vindr_cxr" in exp_configs.dataset or exp_configs.dataset == "chexpert":
        data_loader["BB_test"] = datamodule.BBox_test_dataloader(batch_size=1, shuffle=False)

    if "resnet" in exp_configs.exp_name:
        data_loader["train"] = None
        data_loader['valid'] = None
        data_loader['test'] = None
        if "bcos" in exp_configs.exp_name:
            solver = bcos_resnet_solver(exp_configs, data_loader=data_loader)
        else:
            solver = resnet_solver(exp_configs, data_loader=data_loader)
            solver.set_explainer(which_explainer=exp_configs.attr_method)

    if "attri-net" in exp_configs.exp_name:
        data_loader['train_pos'] = None
        data_loader['train_neg'] = None
        data_loader['vis_pos'] = None
        data_loader['vis_neg'] = None
        data_loader['valid'] = None
        data_loader['test'] = None
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
    analyser = pixel_sensitivity_analyser(solver, config)
    results = analyser.run()

    return results


if __name__ == "__main__":

    # evaluated_models = guided_attrinet_models
    # file_name = str(datetime.datetime.now())[:-7] + "_eval_pixel_sensitivity_" + "guided_attrinet_models" + ".json"

    # evaluated_models = attrinet_models
    # file_name = str(datetime.datetime.now())[
    #             :-7] + "_eval_pixel_sensitivity_" + "attrinet_models" + ".json"


    # evaluated_models = guided_bcos_resnet_models
    # file_name = str(datetime.datetime.now())[
    #             :-7] + "_eval_pixel_sensitivity_" + "guided_bcos_resnet_models" + ".json"


    evaluated_models = bcos_resnet_models
    file_name = str(datetime.datetime.now())[
                :-7] + "_eval_pixel_sensitivity_" + "bcos_resnet_models" + ".json"

    # out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/tmi_results"
    out_dir = "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/tmi_results/revision_20250625"

    parser = argument_parser()
    opts = parser.parse_args()

    if "resnet" in file_name and "bcos" not in file_name:
        for explanation_method in ['gifsplanation']:
            results_dict = {}
            for key, value in evaluated_models.items():
                model_path = value
                print("Now evaluating model: " + model_path)
                opts.model_path = model_path
                opts.attr_method = explanation_method
                opts = update_params_with_model_path(opts, model_path)
                results = main(opts)
                results_dict[key + "_" +explanation_method] = results
            print(results_dict)
            output_path = os.path.join(out_dir, file_name+"_"+explanation_method)
            with open(output_path, 'w') as json_file:
                json.dump(results_dict, json_file, indent=4)

    else:
        results_dict = {}
        for key, value in evaluated_models.items():
            model_path = value
            print("Now evaluating model: " + model_path)
            opts.model_path = model_path
            opts = update_params_with_model_path(opts, model_path)
            results = main(opts)
            results_dict[key] = results

        print(results_dict)
        output_path = os.path.join(out_dir, file_name)
        with open(output_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
