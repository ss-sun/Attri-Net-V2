import os
import numpy as np
import torch
import random
import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
# from solvers.attrinet_solver import task_switch_solver
# from solvers.attrinet_solver_energyloss_new import task_switch_solver
from solvers.attrinet_solver import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from train_utils import to_numpy
from tqdm import tqdm
from PIL import Image, ImageDraw
from model_dict import resnet_models, bcos_resnet_models, attrinet_models, aba_loss_attrinet_models
import datetime
from eval_utils import get_weighted_map, vis_samples
import json



def str2bool(v):
    return v.lower() in ('true')


class class_sensitivity_analyser():
    def __init__(self, solver, config):
        self.dataset = config.dataset  # different dataset has differnt annotation dataframe format.
        self.solver = solver
        self.result_dir = os.path.join(config.model_path, "class_sensitivity_result_dir")
        self.train_diseases = self.solver.TRAIN_DISEASES
        self.best_threshold = {}
        self.attr_method = config.attr_method
        os.makedirs(self.result_dir, exist_ok=True)
        self.attr_dir = os.path.join(self.result_dir, self.attr_method)
        os.makedirs(self.attr_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.result_dir, self.attr_method + '_class_sensitivity_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        self.draw = True
        print('compute class sensitivity on dataset ' + config.dataset)
        print('use explainer: ' + self.attr_method)

        # Read data from file.
        threshold_path = os.path.join(self.result_dir, "best_threshold.txt")
        if os.path.exists(threshold_path):
            print("threshold alreay computed, load threshold")
            # threshold = np.loadtxt(open(threshold_path)).reshape(1,)
            threshold = np.loadtxt(open(threshold_path))
            for c in range(len(self.solver.TRAIN_DISEASES)):
                disease = self.solver.TRAIN_DISEASES[c]
                self.best_threshold[disease] = threshold[c]

        else:
            self.best_threshold = solver.get_optimal_thresholds(save_result=True, result_dir=self.result_dir)

        pred_file = os.path.join(self.result_dir,"test_pred.txt")
        if os.path.exists(pred_file):
            print("prediction of test set already made")
            self.test_pred = np.loadtxt(os.path.join(self.result_dir,"test_pred.txt"))
            self.test_true = np.loadtxt(os.path.join(self.result_dir,"test_true.txt"))
        else:
            self.test_pred, self.test_true, class_auc = self.solver.test(which_loader="test", save_result=True, result_dir=self.result_dir)
            print('test_class_auc', class_auc)

    def filter_correct_pred(self, test_pred, test_true, train_disease, best_threshold):
        test_pred = test_pred.reshape(test_pred.shape[0],-1)
        test_true = test_true.reshape(test_true.shape[0],-1)
        pred = np.zeros(test_pred.shape)
        for i in range(len(train_disease)):
            disease = train_disease[i]
            pred[np.where(test_pred[:, i] > best_threshold[disease])[0], i] = 1
            pred[np.where(test_pred[:, i] < best_threshold[disease])[0], i] = 0
        results = {}
        for i in range(len(train_disease)):
            r = {}
            disease = train_disease[i]
            pos_idx = np.where(pred[:, i] == 1)[0]
            neg_idx = np.where(pred[:, i] == 0)[0]
            t_pos_idx = np.where(test_true[:, i] == 1)[0]
            t_neg_idx = np.where(test_true[:, i] == 0)[0]
            TP = list(set(pos_idx.tolist()).intersection(t_pos_idx.tolist()))
            TN = list(set(neg_idx.tolist()).intersection(t_neg_idx.tolist()))
            tp_pred = np.column_stack((TP, test_pred[:,i][TP]))
            tn_pred = np.column_stack((TN, test_pred[:,i][TN]))
            sorted_tp_pred = tp_pred[tp_pred[:, 1].argsort()] # from smallest to biggest
            sorted_tn_pred = tn_pred[tn_pred[:, 1].argsort()] # from smallest to biggest
            sorted_TP = sorted_tp_pred[:,0]
            sorted_TN = sorted_tn_pred[:,0]
            r['TP'] = sorted_TP[::-1] # from biggest to smallest
            r['TN'] = sorted_TN # from smallest to biggest
            results[disease] = r
        return results

    def create_grids(self, n_cells, pred_dict, train_disease, num_imgs):
        # n_cells is 2 or 3. to make grid 2*2 or 3*3
        grids = {}
        num_neg = n_cells * n_cells - 1
        for disease in train_disease:
            blocks = []
            preds = pred_dict[disease]
            TP = preds['TP'][:num_imgs].tolist()
            TN = preds['TN'][:num_imgs*num_neg].tolist()

            random.shuffle(TN)
            current_point = 0
            for idx in TP:
                b = []
                b.append(idx)
                if (current_point+num_neg) <= len(TN):
                    for j in range(current_point, current_point+num_neg):
                        b.append(TN[j])
                    current_point += num_neg
                else:
                    current_point=0
                blocks.append(b)
            grids[disease] = blocks
        return grids

    def compute_localization_score(self, idx_grids, attr_method):
        scores = {}
        mean = 0
        for disease in idx_grids.keys():
            self.draw = True
            score_list = []
            blocks = idx_grids[disease]
            for i in tqdm(range(len(blocks))):
                if i > 10:
                    self.draw = False
                b = blocks[i]
                sc = self.compute_sc(i, b, disease, attr_method)
                score_list.append(sc)

            avg_score = np.mean(np.array(score_list))
            scores[disease] = str(avg_score)
            mean += avg_score

        mean = mean / (len(idx_grids.keys()))
        print('mean localization score on all disease: ', mean)

        return scores

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




    def compute_sc(self, rank, index_list, disease, attr_method):

        label_idx = self.solver.TRAIN_DISEASES.index(disease)
        pixel_counts = []
        count = 0
        for idx in index_list:
            data = self.solver.test_loader.dataset[int(idx)]
            img_id = self.get_img_id(int(idx))
            img = data['img']
            img = torch.from_numpy(img[None])

            if attr_method in ['attri-net']:
                dests, attr_raw = self.solver.get_attributes(img, label_idx)
                if attr_raw.shape[1] == 3:
                    attr_raw = torch.mean(attr_raw, dim=1, keepdim=True)

                weighted_map = get_weighted_map(to_numpy(attr_raw).squeeze(), lgs=self.solver.net_lgs[disease])
                # after multiplying with weighted map, the positive values in weighted map means postive contribution and negative values means negative contribution.
                attr = np.where(weighted_map > 0, weighted_map, 0)
                sum_pixel = np.sum(attr)

                # attr = to_numpy(attr_raw).squeeze()
                # sum_pixel = np.sum(abs(attr))

            else:
                attr = self.solver.get_attributes(img, label_idx)
                dests = None
                attr = to_numpy(attr).squeeze()
                sum_pixel = np.sum(attr)

            if self.draw:
                prefix = 'rank' + str(rank) + '_' + str(count) + '_' + img_id + '_' + attr_method + '_' + disease
                vis_samples(src_img=img, attr=attr, dests=dests, prefix=prefix,
                            output_dir=self.plots_dir, attr_method=self.attr_method)
            count += 1
            pixel_counts.append(sum_pixel)

        if np.sum(np.array(pixel_counts)) !=0 :
            score = pixel_counts[0] / (np.sum(np.array(pixel_counts)))
        else:
            score = 0
        return score


    def run(self):
        # Filter truly predicted images
        filter_results = self.filter_correct_pred(self.test_pred, self.test_true, train_disease=self.train_diseases, best_threshold=self.best_threshold)
        # Creating 2*2 blocks
        idx_grids = self.create_grids(2, filter_results, self.train_diseases, num_imgs=200)
        # Compute class sensitivity score.
        local_score = self.compute_localization_score(idx_grids, attr_method=self.attr_method)
        print(local_score)
        return local_score







def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='attri-net', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--attr_method', type=str, default='lime',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'attri-net' , 'gifsplanation', 'bcos'")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--img_mode', type=str, default='gray',
                        choices=['color', 'gray'])  # will change to color if dataset is airogs_color

    parser.add_argument('--dataset', type=str, default='chexpert',
                        choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'contaminated_chexpert'])
    parser.add_argument("--batch_size", default=4,
                        type=int, help="Batch size for the data loader.")
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
    if "resnet" in exp_configs.exp_name:
        data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        if "bcos" in exp_configs.exp_name:
            solver = bcos_resnet_solver(exp_configs, data_loader=data_loader)
        else:
            solver = resnet_solver(exp_configs, data_loader=data_loader)
            solver.set_explainer(which_explainer=exp_configs.attr_method)

    if "attri-net" in exp_configs.exp_name:
        train_loaders = datamodule.single_disease_train_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)
        vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=exp_configs.batch_size, shuffle=False)
        valid_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader['train_pos'] = train_loaders['pos']
        data_loader['train_neg'] = train_loaders['neg']
        data_loader['vis_pos'] = vis_dataloaders['pos']
        data_loader['vis_neg'] = vis_dataloaders['neg']
        data_loader['valid'] = valid_loader
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
    analyser = class_sensitivity_analyser(solver, config)
    results = analyser.run()
    return results


if __name__ == "__main__":
    # set the variables here:
    evaluated_models = aba_loss_attrinet_models
    file_name = str(datetime.datetime.now())[:-7] + "eval_class_sensitivity_" + "aba_loss_attrinet_models" + ".json"

    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/tmi_results"

    parser = argument_parser()
    opts = parser.parse_args()

    if "resnet" in file_name and "bcos" not in file_name:
        for explanation_method in ["shap", "gifsplanation"]:
            results_dict = {}
            for key, value in evaluated_models.items():
                model_path = value
                print("Now evaluating model: " + model_path)
                opts.model_path = model_path
                opts.attr_method = explanation_method
                opts = update_params_with_model_path(opts, model_path)
                results = main(opts)
                results_dict[key+"_"+explanation_method] = results
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