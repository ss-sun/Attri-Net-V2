import argparse
from train_utils import prepare_datamodule


def str2bool(v):
    return v.lower() in ('true')

def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='resnet', choices=['resnet', 'attri-net'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='vindr_cxr', choices=['chexpert', 'nih_chestxray', 'vindr_cxr'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    return parser












# Description: This file contains the paths to the trained models
model_dict = {
    "attri-net_previous_aug": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    "attri-net_all_aug": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:03--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=all--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    "attri-net_color_aug": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:42:26--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=color_jitter--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    "attri-net_centercrop_aug": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:42:00--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=center_crop--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    "attri-net_randomcrop_aug": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:40:34--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=random_crop--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
}


# load model


# load test data


# use mean of all models.
def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)