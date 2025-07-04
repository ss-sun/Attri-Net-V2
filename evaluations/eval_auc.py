import sys
import os
# sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/github_projects/tmi"))

sys.path.append(os.path.abspath("/home/susu/projects/AttriNet_revision/Attri-Net-V2"))

import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from model_dict import resnet_models, bcos_resnet_models, attrinet_models, aba_loss_attrinet_models, aba_guidance_attrinet_models, guided_attrinet_models, guided_bcos_resnet_models
import json
import datetime
import time



def str2bool(v):
    return v.lower() in ('true')

class classification_analyser():
    """
    Analyse the classification performance of the model.
    """
    def __init__(self, solver):
        self.solver = solver

    def run(self):
        start_time = time.time()
        _, _, auc = self.solver.test()
        end_time = time.time()
        print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
        num_samples = self.solver.test_loader.dataset.__len__()
        print(num_samples)
        avg_time_per_sample = (end_time - start_time) / num_samples
        print(f"Average time per sample: {avg_time_per_sample:.4f} seconds.")
        return auc

def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='attri-net', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--img_mode', type=str, default='gray', choices=['color', 'gray']) # will change to color if dataset is airogs_color
    # Data configuration.
    parser.add_argument('--dataset', type=str, default='nih_chestxray', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'contaminated_chexpert'])
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
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



def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    solver = prep_solver(datamodule, config)
    analyser = classification_analyser(solver)
    result = analyser.run()
    return result


def update_params_with_model_path(opts, model_path):
    if 'attri-net' in model_path:
        opts.exp_name = 'attri-net'
        opts = update_attrinet_params(opts)
    if 'resnet' in model_path:
        opts.exp_name = 'resnet'
    if 'bcos' in model_path:
        opts.exp_name = 'bcos_resnet'
    if 'chexpert' in model_path:
        opts.dataset = 'chexpert'
    if 'nih' in model_path:
        opts.dataset = 'nih_chestxray'
    if 'vindr' in model_path:
        opts.dataset = 'vindr_cxr'

    print("evaluating model: " + opts.exp_name + " on dataset: " + opts.dataset)

    return opts



if __name__ == "__main__":
    eval_on_OOD = True  # Set to True if you want to evaluate on OOD datasets

    if eval_on_OOD == True:
        which_model = "bcos_resnet_models"

        #### evaluate model on slighly different datasets ####, model trained on chexpert, but evaluated on nih_chestxray.
        model_dict = {
            "resnet_models": {
                "chexpert": "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/resnet/resnet2023-12-07 10:03:30-nih_chestxray-bs=4-lr=0.0001-weight_decay=1e-05",
                "nih_chestxray": "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/resnet/resnet2023-12-07 10:14:57-chexpert-bs=4-lr=0.0001-weight_decay=1e-05",
            },
            "bcos_resnet_models": {
                "chexpert": "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/bcos_resnet/bcos_resnet2023-12-07 16:20:17-nih_chestxray-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0",
                "nih_chestxray": "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/bcos_resnet/bcos_resnet2023-12-07 16:44:27-chexpert-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0",
            },
            "attrinet_models": {
                "chexpert": "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/attri-net/attri-net2023-12-08 17:29:36--nih_chestxray--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
                "nih_chestxray": "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/attri-net/attri-net2023-12-08 17:29:36--chexpert--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
            },
            "attrinet_guided_models": {
                "chexpert":"/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/attri-net/attri-net2023-12-11 15:40:33--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42",
                "nih_chestxray":"/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42",
            },
        }
        file_name = str(datetime.datetime.now())[:-7] + "eval_auc_" + which_model+"_ood" + ".json"
        out_dir = "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/tmi_results"
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir, exist_ok=True)

        parser = argument_parser()
        opts = parser.parse_args()
        results_dict = {}
        evaluated_models = model_dict[which_model]
        for key, value in evaluated_models.items():
            opts.dataset = key
            opts.model_path = value
            print("evaluating model: " + opts.model_path)
            if 'attri-net' in opts.model_path:
                opts.exp_name = 'attri-net'
                opts = update_attrinet_params(opts)
            if 'resnet' in opts.model_path:
                opts.exp_name = 'resnet'
            if 'bcos' in opts.model_path:
                opts.exp_name = 'bcos_resnet'
            print("evaluating model: " + opts.exp_name + " on dataset: " + opts.dataset)


            # opts = update_params_with_model_path(opts, model_path)
            results = main(opts)
            results_dict[key] = results
            print(results_dict)

        output_path = os.path.join(out_dir, file_name)
        with open(output_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)




    if eval_on_OOD == False:

        # set the variables here:
        # evaluated_models = resnet_models
        # file_name = str(datetime.datetime.now())[:-7] + "eval_auc_" + "resnet_models" + ".json"

        # evaluated_models = bcos_resnet_models
        # file_name = str(datetime.datetime.now())[:-7] + "eval_auc_" + "bcos_resnet_models" + ".json"


        # evaluated_models = guided_attrinet_models
        # file_name = str(datetime.datetime.now())[:-7] + "eval_auc_" + "guided_attrinet_models" + ".json"

        # evaluated_models = attrinet_models
        # file_name = str(datetime.datetime.now())[:-7] + "eval_auc_" + "attrinet_models" + ".json"

        evaluated_models = guided_bcos_resnet_models
        file_name = str(datetime.datetime.now())[:-7] + "eval_auc_" + "guided_bcos_resnet_models" + ".json"

        # set above variables
        # out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/tmi_results"
        out_dir = "/mnt/lustre/work/baumgartner/sun22/exps/TMI_exps/tmi_results"
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir,exist_ok=True)

        parser = argument_parser()
        opts = parser.parse_args()

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
