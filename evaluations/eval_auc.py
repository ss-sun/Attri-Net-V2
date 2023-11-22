import sys
import os
sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/project/tmi"))

import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
# from solvers.attrinet_solver import task_switch_solver
from solvers.attrinet_solver_energyloss_new import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from model_dict import resnet_model_path_dict, attrinet_model_path_dict, bcos_resnet_model_path_dict, \
    attrinet_vindrBB_different_lambda_dict, bcos_vindr_with_guidance_dict, bcos_chexpert_with_guidance_dict, \
    bcos_nih_chestxray_with_guidance_dict, attrinet_chexpert_with_guidance_dict, attrinet_nih_chestxray_with_guidance_dict,\
    attrinet_vindr_cxr_withBB_with_guidance_dict, glaucoma_dict
import json
import datetime


def str2bool(v):
    return v.lower() in ('true')

class classification_analyser():
    """
    Analyse the classification performance of the model.
    """
    def __init__(self, solver):
        self.solver = solver

    def run(self):
        _, _, auc = self.solver.test()
        return auc

def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--debug', type=str2bool, default=False, help='if true, print more informatioin for debugging')
    parser.add_argument('--exp_name', type=str, default='bcos_resnet', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--img_mode', type=str, default='gray',
                        choices=['color', 'gray'])  # will change to color if dataset is airogs_color
    parser.add_argument('--process_mask', type=str, default='previous', choices=['abs(mx)', 'sum(abs(mx))', 'previous'])
    # Data configuration.
    # parser.add_argument('--dataset', type=str, default='airogs', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea', 'airogs', 'airogs_color' ,'vindr_cxr_withBB', 'contam20', 'contam50'])
    parser.add_argument('--dataset_idx', type=int, default=6,
                        help='index of the dataset in the datasets list, convinent for submitting parallel jobs')

    # parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
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


if __name__ == "__main__":

    # set the variables here:

    evaluated_models = bcos_vindr_with_guidance_dict
    # evaluated_models = {}
    # evaluated_models["vagan_color"] = glaucoma_dict["vagan_color"]

    file_name = str(datetime.datetime.now())[:-7] +"eval_auc_"+"bcos_vindr_with_guidance_dict"+".json"

    # set above variables

    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results"
    parser = argument_parser()
    opts = parser.parse_args()
    datasets = ['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea', 'airogs', 'airogs_color', 'vindr_cxr_withBB',
                'contam20', 'contam50']
    opts.dataset = datasets[opts.dataset_idx]
    if 'color' in opts.dataset:
        opts.img_mode = 'color'

    results_dict = {}
    for key, value in evaluated_models.items():
        model_path = value
        print("Now evaluating model: " + model_path)
        assert opts.dataset in model_path
        opts.model_path = model_path
        if 'attri-net' in model_path:
            opts = update_attrinet_params(opts)
        results = main(opts)
        results_dict[key] = results



    print(results_dict)


    output_path = os.path.join(out_dir, file_name)
    with open(output_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
