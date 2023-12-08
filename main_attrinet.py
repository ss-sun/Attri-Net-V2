from solvers.attrinet_solver import task_switch_solver
from experiment_utils import init_seed, init_experiment, init_wandb
from train_utils import prepare_datamodule

import argparse

def str2bool(v):
    return v.lower() in ('true')

def attrinet_get_parser():
    parser = argparse.ArgumentParser()

    # Experiment configuration.
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='if true, will automatically set d_iters = 1, set savefrequency=1, easy to run all train step for functional test')

    parser.add_argument('--exp_name', type=str, default='attri-net')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    parser.add_argument('--guidance_mode', type=str, default='no_guidance',
                        choices=['bbox/masks', 'pseudo_mask', 'mixed', 'no_guidance'])  # use bbox or pseudo_mask as guidance of disease mask for better localization.

    parser.add_argument('--guidance_freq', type=float, default=0.1, help='frequency to train with BBox')
    # Data configuration.
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'contaminated_chexpert'])

    parser.add_argument('--image_size', type=int, default=320, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')

    # Model configuration.
    # Configurations of latent code generator
    parser.add_argument('--n_fc', type=int, default=8, help='number of fc layer in Intermediate_Generator inside generator')
    parser.add_argument('--n_ones', type=int, default=20, help='number of ones to indicting each task, will affect the latent dim of task vector in generator,default is 20')
    parser.add_argument('--num_out_channels', type=int, default=1, help='number of out channels of generator')

    # Configurations of logistic regression classifier
    parser.add_argument('--lgs_downsample_ratio', type=int, default=32,
                        help='downsampling ratio of logistic regression classifier, can be 4, 8, 16, 32, 64, 80, 160')

    # Configurations of generator
    parser.add_argument('--lambda_critic', type=float, default=1.0, help='weight for critic loss')
    parser.add_argument('--lambda_1', type=float, default=100, help='weight for l1 loss of disease mask')
    parser.add_argument('--lambda_2', type=float, default=200, help='weight for l1 loss of healthy mask')
    parser.add_argument('--lambda_3', type=float, default=100, help='weight for classification loss')
    parser.add_argument('--lambda_centerloss', type=float, default=0.01, help='weight for center loss of disease mask')

    parser.add_argument('--lambda_localizationloss', type=float, default=0, help='weight for localization loss of disease mask, default=30')

    # Training configuration.
    parser.add_argument('--cls_iteration', type=int, default=5, help='number of classifier iterations per each generator iter, default=5')
    parser.add_argument('--d_iters', type=int, default=5, help='number of discriminator iterations per each generator iter, default=5')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training generator')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lgs_lr', type=float, default=0.0001, help='learning rate for logistic regression classifier, previous exp use 0.00025')
    parser.add_argument('--weight_decay_lgs', type=float, default=0.00001, help='weight decay for logistic regression classifier')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam, default 0.9')
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--save_path', type=str, default='/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net',
                        help='path of the exp')

    # Step size.
    parser.add_argument('--sample_step', type=int, default=1000,
                        help='frequency of saving visualization samples, default = 500')
    parser.add_argument('--model_valid_step', type=int, default=1000, help='frequency of validation')
    parser.add_argument('--lr_update_step', type=int, default=1000, help='frequency of learning rate update')

    # Testing configuration.
    parser.add_argument('--test_model_path', type=str, default=None, help='path of the models')

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
    data_loader = {}
    print("working on attri-net")
    train_loaders = datamodule.single_disease_train_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)
    vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=4, shuffle=False)
    valid_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
    test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)

    data_loader['train_pos'] = train_loaders['pos']
    data_loader['train_neg'] = train_loaders['neg']
    data_loader['vis_pos'] = vis_dataloaders['pos']
    data_loader['vis_neg'] = vis_dataloaders['neg']
    data_loader['valid'] = valid_loader
    data_loader['test'] = test_loader

    if exp_configs.dataset == "nih_chestxray":
        data_loader['train_pos_bbox'] = datamodule.single_disease_trainBBox_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)

    solver = task_switch_solver(exp_configs, data_loader=data_loader)

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
    parser = attrinet_get_parser()
    config = parser.parse_args()
    main(config)