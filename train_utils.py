import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
# from data.chexpert import CheXpertDataModule
from data.chexpert_official_split import CheXpertDataModule
from data.nih_chestxray import NIHChestXrayDataModule
from data.vindr_cxr import Vindr_CXRDataModule
from data.skm_tea_mri import SKMTEADataModule
from data.contaminated_chexpert import Contaminate_CheXpertDataModule
from data.airogs_glaucoma import AIROGS_fundusDataModule
from data.airogs_glaucoma_color import AIROGS_color_fundusDataModule
from data.vindr_cxr_withBB import Vindr_CXR_BB_DataModule


def prepare_datamodule(exp_configs, dataset_dict, data_default_params):
    # prepare dataloaders
    dataset_params = dataset_dict[exp_configs.dataset]
    exp_configs.train_diseases = dataset_params["train_diseases"]
    if exp_configs.dataset == 'chexpert':
        print("working on chexpert dataset")
        # datamodule = CheXpertDataModule(dataset_params,
        #                                 split_ratio=data_default_params['split_ratio'],
        #                                 resplit=data_default_params['resplit'],
        #                                 img_size=data_default_params['img_size'],
        #                                 seed=exp_configs.manual_seed)
        datamodule = CheXpertDataModule(dataset_params,
                                        img_size=data_default_params['img_size'],
                                        seed=exp_configs.manual_seed)  # use official split
    if exp_configs.dataset == 'nih_chestxray':
        print("working on nih_chestxray dataset")
        datamodule = NIHChestXrayDataModule(dataset_params,
                                            split_ratio=data_default_params['split_ratio'],
                                            resplit=data_default_params['resplit'],
                                            img_size=data_default_params['img_size'],
                                            seed=exp_configs.manual_seed)
    if exp_configs.dataset == 'vindr_cxr':
        print("working on vindr_cxr dataset")
        datamodule = Vindr_CXRDataModule(dataset_params,
                                         split_ratio=data_default_params['split_ratio'],
                                         resplit=data_default_params['resplit'],
                                         img_size=data_default_params['img_size'],
                                         seed=exp_configs.manual_seed)

    if exp_configs.dataset == 'skmtea':
        print("working on skmtea-mri dataset")
        datamodule = SKMTEADataModule(dataset_params,
                                        img_size=data_default_params['img_size'],
                                        seed=exp_configs.manual_seed)  # use official split
    if exp_configs.dataset == 'airogs':
        datamodule = AIROGS_fundusDataModule(dataset_params,
                                             img_size=data_default_params['img_size'],
                                             seed=42)

    if exp_configs.dataset == 'airogs_color':
        datamodule = AIROGS_color_fundusDataModule(dataset_params,
                                             img_size=data_default_params['img_size'],
                                             seed=42)

    if 'contam' in exp_configs.dataset:
        print("working on contaminated dataset")
        datamodule = Contaminate_CheXpertDataModule(dataset_params,
                                      img_size=data_default_params['img_size'],
                                      seed=exp_configs.manual_seed)  # use official split


    if 'vindr_cxr_withBB' in exp_configs.dataset:
        print("working on vindr_cxr_withBB dataset")
        datamodule = Vindr_CXR_BB_DataModule(dataset_params,
                                             split_ratio=data_default_params['split_ratio'],
                                             resplit=data_default_params['resplit'],
                                             img_size=data_default_params['img_size'],
                                             seed=42)

    datamodule.setup()
    return datamodule





def print_network(model):
    """
    Print out the network information.
    """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

def logscalar(name, value):
    wandb.log({name: value})


def to_numpy(tensor):
    """
    Converting tensor to numpy.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


def save_batch(img_batch, label_batch, pred_batch=None, out_dir=''):

    vmax = np.abs(img_batch).flatten().max()
    vmin = np.abs(img_batch).flatten().min()

    cols = int(img_batch.shape[0] / 2)
    rows = 2
    figure = plt.figure(figsize=(5*cols, 3*rows))

    for i in range(1, cols * rows + 1):
        img = img_batch[i-1]
        label = label_batch[i-1]
        figure.add_subplot(rows, cols, i)
        title = 'label: ' + str(label)
        if pred_batch is not None:
            pred = pred_batch[i-1]
            title = title + '  pred: '+ str(pred.squeeze())[:4]
        plt.title(title)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)

    plt.savefig(out_dir, bbox_inches='tight')

