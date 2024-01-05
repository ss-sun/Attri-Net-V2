from solvers.bcosnet_solver import bcos_resnet_solver
from experiment_utils import init_seed, init_experiment, init_wandb
from train_utils import prepare_datamodule
import argparse

def str2bool(v):
    return v.lower() in ('true')


def bcos_resnet_get_parser():
    parser = argparse.ArgumentParser()

    # Experiment configuration.
    parser.add_argument('--debug', type=str2bool, default=False, help='if true, print more informatioin for debugging')

    parser.add_argument('--exp_name', type=str, default='bcos_resnet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    parser.add_argument('--guidance_mode', type=str, default="mixed",
                        choices=['no_guidance', 'full_guidance', 'mixed', 'mixed_weighted', 'full'])  # use bbox or pseudo_mask as guidance of disease mask for better localization.


    # Data configuration.

    parser.add_argument('--dataset', type=str, default='nih_chestxray', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'vindr_cxr_mix', 'chexpert_mix',])

    parser.add_argument('--image_size', type=int, default=320, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument("--lambda_localizationloss", type=float, default=0.1, help="Lambda to use to weight localization loss. 0 means no localization loss.")
    # Training configuration.
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--save_path', type=str, default='/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet', help='path of the exp')

    # Testing configuration.
    parser.add_argument('--test_model_path', type=str, default='/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet', help='path of the models')

    # Miscellaneous.
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')

    return parser



def prepare_exps(exp_configs):
    if exp_configs.mode == 'train':
        print("training model: ")
        init_experiment(exp_configs)
    init_seed(exp_configs.manual_seed)
    if exp_configs.use_wandb:
        init_wandb(exp_configs)


def main(exp_configs):
    from data.dataset_params import dataset_dict, data_default_params

    prepare_exps(exp_configs)
    print("experiment folder: " + exp_configs.exp_dir)

    datamodule = prepare_datamodule(exp_configs, dataset_dict, data_default_params)
    print(exp_configs)

    # Prepare data loaders and solver.
    print("working on bcos_resnet")
    data_loader = {}
    data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
    data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
    data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)

    if (exp_configs.guidance_mode in ['mixed', 'mixed_weighted']):
        assert exp_configs.dataset in ['nih_chestxray', 'chexpert_mix', 'vindr_cxr_mix']
        data_loader['train_pos_bbox'] = datamodule.trainBBox_dataloader(batch_size=exp_configs.batch_size, shuffle=True)

    solver = bcos_resnet_solver(exp_configs, data_loader=data_loader)


    if exp_configs.mode == "train":
        print('start training...')
        solver.train()
        print('finish training!')

    if exp_configs.mode == 'test':
        print('start testing....')
        solver.load_model(exp_configs.test_model_path)
        test_auc = solver.test()
        print('finish test!')
        print('test_auc: ', test_auc)





if __name__ == '__main__':
    parser = bcos_resnet_get_parser()
    config = parser.parse_args()
    main(config)