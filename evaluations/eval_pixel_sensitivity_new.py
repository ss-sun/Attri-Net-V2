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
from solvers.attrinet_solver_energyloss import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from eval_utils import get_weighted_map, draw_BB, draw_hit, vis_samples
from pycocotools import mask
from model_dict import resnet_model_path_dict, attrinet_model_path_dict, bcos_resnet_model_path_dict, attrinet_vindrBB_different_lambda_dict











def str2bool(v):
    return v.lower() in ('true')


class pixel_sensitivity_analyser():
    def __init__(self, solver, config):
        self.config = config
        self.top_k = config.top_k
        self.dataset = config.dataset # different dataset has different gt annotation format
        self.attr_method = config.attr_method
        self.process_mask = config.process_mask
        self.solver = solver
        self.train_diseases = self.solver.TRAIN_DISEASES
        self.result_dir = os.path.join(config.result_dir, self.attr_method)
        os.makedirs(self.result_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.result_dir, self.attr_method + '_pixel_sensitivity_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        self.draw = True
        if self.dataset == "chexpert":
            gt_seg_file = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
            with open(gt_seg_file) as json_file:
                self.gt_seg_dict = json.load(json_file)

    def run(self):
        results = self.compute_pixel_sensitivity(attr_method=self.attr_method)
        return results

    def create_mask_fromBB(self, img_size, bbox):
        #bbox: [x, y, w, h]
        mask = np.zeros(img_size)
        mask[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
        return mask

    def compute_hit_nih_vindr_skmtea(self, attr_method):

        samples_per_disease = {}
        hits_per_disease = {}
        pixel_localization_accuracy = {}
        total = 0
        hits = 0
        for disease in self.solver.TRAIN_DISEASES:
            hits_per_disease[disease] = 0
            samples_per_disease[disease] = 0
            pixel_localization_accuracy[disease] = 0

        for idx, row in self.solver.dataloaders["BB_test"].dataset.df.iterrows():
            if self.dataset == 'nih_chestxray':
                disease = row['Finding Label']
                img_id = row['Image Index'][:-4]
            if self.dataset == 'vindr_cxr':
                disease = row['class_name']
                img_id = row['image_id']
            if self.dataset == 'skmtea':
                disease = row['Finding Label']
                img_id = row['Image Index']

            if disease not in self.solver.TRAIN_DISEASES:
                continue  # means no self.solver.TRAIN_DISEASES in this image
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

                if self.draw:
                    draw_BB(to_numpy(img.squeeze() * 0.5 + 0.5), bbox, img_id, 'gt_input: ' + disease, self.plots_dir)

                if attr_method == 'attri-net':
                    attr = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                else:
                    attr_max = np.max(attr_raw)
                    attr_min = np.min(attr_raw)
                    if attr_max != attr_min:
                        attr = (attr_raw - attr_min) / (attr_max - attr_min)
                    else:
                        attr = attr_raw

                gt_mask = self.create_mask_fromBB(img_size=attr.shape, bbox=bbox)
                hit, pos = self.get_hit_topK(attr_map=attr, gt_annotation=gt_mask, top_k=self.top_k)

                hits += hit
                hits_per_disease[disease] += hit
                prefix = img_id + '_hit_' + str(hit) + '_' + attr_method + '_' + disease
                if pos != np.nan and self.draw:
                    draw_hit(pos, attr, gt_annotation=bbox, prefix=prefix, output_dir=self.plots_dir, gt="bbox")

        for disease in self.solver.TRAIN_DISEASES:
            if samples_per_disease[disease] != 0:
                pixel_localization_accuracy[disease] = hits_per_disease[disease] / samples_per_disease[disease]
        print('hits/total ', hits/total)
        print("samples_per_disease: ", samples_per_disease)
        print("hits_per_disease: ", hits_per_disease)
        print("pixel_localization_accuracy: ", pixel_localization_accuracy)
        result_file_path = os.path.join(self.result_dir, "pixel_sensitivity_results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump([ {"hits": hits, "total": total, "hits/total": hits/total}, samples_per_disease, hits_per_disease, pixel_localization_accuracy], json_file, indent=4)

        return pixel_localization_accuracy


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


    def compute_EPG_nih_vindr_skmtea(self, attr_method):

        print("Now computing EPG score for: ", self.dataset)
        print("current attr_method: ", attr_method)

        EPG_score = {}
        avg_EPG_score = {}
        for disease in self.solver.TRAIN_DISEASES:
            EPG_score[disease] = []
            avg_EPG_score[disease] = 0

        for idx, row in self.solver.dataloaders["BB_test"].dataset.df.iterrows():
                if self.dataset == 'nih_chestxray':
                    disease = row['Finding Label']
                    img_id = row['Image Index'][:-4]
                if self.dataset == 'vindr_cxr':
                    disease = row['class_name']
                    img_id = row['image_id']

                if disease not in self.solver.TRAIN_DISEASES:
                    continue  # means no self.solver.TRAIN_DISEASES in this image
                else:

                    label_idx = self.solver.TRAIN_DISEASES.index(disease)
                    data = self.solver.dataloaders["BB_test"].dataset[int(idx)]
                    img = data["img"]
                    bbox = data["BBox"]
                    img = torch.from_numpy(img[None])
                    dests, attr_raw = self.solver.get_attributes(img, label_idx)
                    attr_raw = to_numpy(attr_raw).squeeze()

                    if attr_method == 'attri-net':
                        attr = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                    else:
                        attr = attr_raw
                    gt_mask = self.create_mask_fromBB(img_size=attr.shape, bbox=bbox)


                    if self.draw:
                        prefix = img_id + '_' + attr_method + '_' + disease
                        vis_samples(src_img=img, attr=attr, dests = dests, gt_annotation=gt_mask, prefix=prefix, output_dir= self.plots_dir)

                    score = self.get_EPG_score(attr, gt_mask)
                    EPG_score[disease].append(score)

        avg_scores = []
        for disease in self.solver.TRAIN_DISEASES:
            scores = EPG_score[disease]
            if scores != []:
                mean_score = np.mean(np.array(scores))
                avg_EPG_score[disease] = str(mean_score)
                avg_scores.append(mean_score)

        print("avg_EPG_score: ", avg_EPG_score)
        print("avg_EPG accross disease: ", np.mean(np.array(avg_scores)))

        result_file_path = os.path.join(self.result_dir, "avg_EPG_results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump(avg_EPG_score, json_file, indent=4)
        return avg_EPG_score



    def compute_EPG_chexpert(self, attr_method):
        print("Now computing EPG score for CheXpert dataset.")
        print("current attr_method: ", attr_method)

        EPG_score = {}
        avg_EPG_score = {}
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
                            attr_raw = self.solver.get_attributes(img, label_idx)
                            attr_raw = to_numpy(attr_raw).squeeze()

                            # process_mask, choices = ['abs(mx)', 'sum(abs(mx))', 'previous']

                            if attr_method == 'attri-net' and self.process_mask != 'sum(abs(mx))':
                                attr = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                            else:
                                attr = attr_raw
                            score = self.get_EPG_score(attr, gt_mask)
                            EPG_score[disease].append(score)
        avg_scores = []
        for disease in self.solver.TRAIN_DISEASES:
            scores = EPG_score[disease]
            if scores != []:
                mean_score = np.mean(np.array(scores))
                avg_EPG_score[disease] = str(mean_score)
                avg_scores.append(mean_score)

        print("avg_EPG_score: ", avg_EPG_score)
        print("avg_EPG accross disease: ", np.mean(np.array(avg_scores)))

        result_file_path = os.path.join(self.result_dir, "avg_EPG_results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump(avg_EPG_score, json_file, indent=4)




    def compute_hit_chexpert(self, attr_method):

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
                            if self.draw:
                                pass
                            if attr_method == 'attri-net':
                                attr = get_weighted_map(attr_raw, lgs=self.solver.net_lgs[disease])
                            else:
                                attr_max = np.max(attr_raw)
                                attr_min = np.min(attr_raw)
                                if attr_max != attr_min:
                                    attr = (attr_raw - attr_min) / (attr_max - attr_min)
                                else:
                                    attr = attr_raw
                            hit, pos = self.get_hit_topK(attr_map=attr, gt_annotation=gt_mask, top_k=self.top_k)
                            hits += hit
                            hits_per_disease[disease] += hit
                            prefix = cxr_id + '_hit_' + str(hit) + '_' + attr_method + '_' + disease
                            if pos != np.nan and self.draw:
                                draw_hit(pos, attr, gt_annotation=gt_mask, prefix=prefix, output_dir=self.plots_dir, gt="mask")


        for disease in self.solver.TRAIN_DISEASES:
            if samples_per_disease[disease] != 0:
                pixel_localization_accuracy[disease] = hits_per_disease[disease] / samples_per_disease[disease]
        print('hits/total ', hits/total)
        print("samples_per_disease: ", samples_per_disease)
        print("hits_per_disease: ", hits_per_disease)
        print("pixel_localization_accuracy: ", pixel_localization_accuracy)
        result_file_path = os.path.join(self.result_dir, "pixel_sensitivity_results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump([ {"hits": hits, "total": total, "hits/total": hits/total}, samples_per_disease, hits_per_disease, pixel_localization_accuracy], json_file, indent=4)

        return pixel_localization_accuracy





    def compute_pixel_sensitivity(self, attr_method):

        if self.dataset == "nih_chestxray" or "vindr_cxr" in self.dataset or self.dataset == "skmtea":

            # self.compute_hit_nih_vindr_skmtea(attr_method)
            avg_EPG_score = self.compute_EPG_nih_vindr_skmtea(attr_method)


        if self.dataset == "chexpert":
            # self.compute_hit_chexpert(attr_method)
            avg_EPG_score = self.compute_EPG_chexpert(attr_method)

        return avg_EPG_score


    def get_hit_topK(self, attr_map, gt_annotation, top_k=1):

        pixel_importance = attr_map
        idcs = np.argsort(pixel_importance.flatten())  # from smallest to biggest
        idcs = idcs[::-1]  # if we want the order biggest to smallest, we reverse the indices array
        idcs = idcs[:top_k]

        # Compute the corresponding masks for deleting pixels in the given order
        positions = np.array(np.unravel_index(idcs, pixel_importance.shape)).T  # first row, h index, second column, w index
        attri_pos_x = positions[:, 0] # position in row
        attri_pos_y = positions[:, 1] # position in column
        top_attri_pixels = list(zip(attri_pos_x, attri_pos_y))
        top_attri_set = set(top_attri_pixels)

        gt_mask = gt_annotation
        assert (gt_mask.shape == pixel_importance.shape)
        if np.sum(gt_mask) == 0:
            hit = np.nan
            pos = np.nan
        else:
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
        # if hit = nan, means no ground truth in this image
        # if hit = 0, means the top 1 pixel is not the ground truth, return the top 1 pixel position
        # if hit = 1, means the top 1 pixel is in the ground truth, return the top 1 pixel position
        return hit, pos




    def get_EPG_score(self, attr_map, gt_annotation):
        # here we only consider the positive contribution. since attribution maps from attri-net are already weighted, therefore, positive value means positive contribution.
        attr = np.where(attr_map > 0, attr_map, 0)
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
    parser.add_argument('--dataset', type=str, default='vindr_cxr', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea'])
    parser.add_argument('--process_mask', type=str, default='previous', choices=['abs(mx)', 'sum(abs(mx))', 'previous'])
    parser.add_argument('--lambda_localizationloss', type=int, default=10, choices=[1, 5, 10, 20, 100])
    parser.add_argument('--top_k', type=int, default=1, help="top k pixels to be considered as hit")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    parser.add_argument('--use_wandb', type=str2bool, default=False, help='whether to use wandb')
    return parser


def update_arguments():
    #from model_dict import resnet_models, attrinet_models

    if exp_configs.exp_name == 'resnet_cls':
        exp_configs.model_path = resnet_model_path_dict[exp_configs.dataset]
        exp_configs.result_dir = os.path.join(exp_configs.model_path, "pixel_sensitivity_result_dir")

    if exp_configs.exp_name == 'bcos_resnet':
        exp_configs.model_path = bcos_resnet_model_path_dict[exp_configs.dataset]
        exp_configs.result_dir = os.path.join(exp_configs.model_path, "pixel_sensitivity_result_dir")

    if exp_configs.exp_name == 'attri-net':
        print("evaluating our model")
        # exp_configs.model_path = attrinet_model_path_dict[exp_configs.dataset]
        exp_configs.model_path = attrinet_model_path_dict[exp_configs.dataset+"_" + exp_configs.process_mask + "_" + str(exp_configs.lambda_localizationloss)]
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


def update_arguments_evaluate_lambdas(model_name):

    if exp_configs.exp_name == 'attri-net':
        print("evaluating our model")
        # exp_configs.model_path = attrinet_model_path_dict[exp_configs.dataset]
        exp_configs.model_path = attrinet_vindrBB_different_lambda_dict[model_name]
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
    analyser = pixel_sensitivity_analyser(solver, config)
    results = analyser.run()

    return results


if __name__ == "__main__":
    parser = argument_parser()
    exp_configs = parser.parse_args()
    results_dict = {}
    for key, value in attrinet_vindrBB_different_lambda_dict.items():
        model_name = key
        print(model_name)
        params = update_arguments_evaluate_lambdas(model_name)
        results = main(params)
        results_dict[model_name] = results
    print(results_dict)
    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results"
    file_name = "vindr_cxr_bbox_lambda_results.json"
    output_path = os.path.join(out_dir, file_name)

    with open(output_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
