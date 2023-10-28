import sys
import os
import numpy as np

import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from train_utils import to_numpy
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw


def str2bool(v):
    return v.lower() in ('true')

def draw_BBox(img, bbox, appendix, out_dir):
    min_row = bbox[0]
    min_col = bbox[1]
    max_row = bbox[2]
    max_col = bbox[3]

    rgb_img = Image.fromarray(img * 255).convert('RGB')
    draw = ImageDraw.Draw(rgb_img)
    outline_color = (255, 0, 0)
    outline_width = 2
    draw.rectangle([min_row, min_col, max_row, max_col], outline=outline_color, width=outline_width)
    del draw
    path = os.path.join(out_dir, appendix+ '_with_BB.png')
    rgb_img.save(path)









class classification_analyser():
    """
    Analyse the classification performance of the model.
    """
    def __init__(self, solver):
        self.solver = solver
        self.out_dir = os.path.join(self.solver.model_path, "eval_results")
        os.makedirs(self.out_dir, exist_ok=True)


    def run(self):
        _, _, auc = self.solver.test()
        return auc

    def plot_centers(self):
        """
        Visualize the class centers during training.
        """
        out = os.makedirs(os.path.join(self.out_dir, "centers"), exist_ok=True)
        for disease in self.solver.TRAIN_DISEASES:
            loss_module = self.solver.center_losses[disease]
            neg_center = loss_module.centers[0].data
            pos_center = loss_module.centers[1].data
            neg_center = to_numpy(neg_center).reshape((self.solver.img_size, self.solver.img_size))
            pos_center = to_numpy(pos_center).reshape((self.solver.img_size, self.solver.img_size))
            filename = disease + "_centers.png"
            out_dir = os.path.join(out, filename)
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4 * 1))
            axs[0].imshow(neg_center, cmap="gray")
            title = "neg center of " + disease
            axs[0].set_title(title)
            axs[0].axis('off')
            axs[1].imshow(pos_center, cmap="gray")
            title = "pos center of " + disease
            axs[1].set_title(title)
            axs[1].axis('off')
            plt.savefig(out_dir, bbox_inches='tight')

    def resize_bbox_mask(self, bbox_mask_orig, tagt_size): # tagt_size: (h, w)
        resized_mask = np.zeros(tagt_size)
        nonzero_indices = np.where(bbox_mask_orig != 0)
        # plt.imshow(Image.fromarray(bbox_mask_orig*255), cmap='gray')
        # plt.show()
        if len(nonzero_indices[0]) != 0:
            # Calculate the bounding box coordinates
            min_row, min_col = np.min(nonzero_indices, axis=1)
            max_row, max_col = np.max(nonzero_indices, axis=1)
            scale_row = tagt_size[0] / bbox_mask_orig.shape[0]
            scale_col = tagt_size[1] / bbox_mask_orig.shape[1]
            min_row = int(min_row * scale_row)
            min_col = int(min_col * scale_col)
            max_row = int(max_row * scale_row)
            max_col = int(max_col * scale_col)
            resized_mask[min_row:max_row, min_col:max_col] = 1
        # plt.imshow(Image.fromarray(resized_mask*255), cmap='gray')
        # plt.show()

        return resized_mask






    def run_withBB(self):
        bbox_dir = "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test"
        superIDtoid = {
            1:[1,2,3,4,5,6,7,8],
            2:[9,10,11],
            3:[12,13,14,15],
            4:[16]
        } #disease supercategory to catergory

        count = {}
        # create folder for each disease
        for disease in self.solver.TRAIN_DISEASES:
            path= os.path.join(self.out_dir, disease)
            os.makedirs(path, exist_ok=True)
            for lbl in ['pos', 'neg']:
                lbl_path = os.path.join(path, lbl)
                os.makedirs(lbl_path, exist_ok=True)
                count[disease + '_' + lbl] = 0

        self.df = self.solver.test_loader.dataset.df
        self.testset = self.solver.test_loader.dataset
        current_scan = None # use current_scan to load bbox, avoid loading bbox for each slice to save time.
        current_bbox = {}
        for idx in np.arange(len(self.df)):
            img_id = self.df.iloc[idx]['image_id'] # print(img_id) # MTR_005_000
            scan_id = img_id[:-4]
            if scan_id != current_scan:
                # reload bbox if scan_id changes.
                print("Now loading bbox for scan: " + scan_id)
                current_scan = scan_id
                print("current_scan", current_scan)
                current_bbox = {}
                current_bbox_dir = os.path.join(bbox_dir, current_scan)
                if os.path.exists(current_bbox_dir):
                    bbox_lbls = [f for f in os.listdir(current_bbox_dir) if f.endswith('.npy')]
                    for bbox_lbl in bbox_lbls:
                        bbox_lbl_path = os.path.join(current_bbox_dir, bbox_lbl)
                        bbox_lbl = int(bbox_lbl[:-4])
                        current_bbox[bbox_lbl] = np.load(bbox_lbl_path)

            if current_bbox != {}:
                # use the same bbox data.
                slice_id = int(img_id[-3:])
                # original_img= np.load(img_path)
                data = self.testset[idx]
                img = data['img']
                lbl = data['label']
                if np.sum(lbl) > 0:
                    values_list = list(count.values())
                    print(count)
                    if np.min(values_list) > 10:
                        break
                    print(img_id)
                    print(lbl)
                    pos_indices = np.where(lbl == 1)[0]
                    for label_idx in pos_indices:
                        img = torch.from_numpy(img[None])
                        attr = self.solver.get_attributes(img, label_idx)
                        attr = to_numpy(attr).squeeze()
                        img = to_numpy(img).squeeze()
                        # plt.imshow(attr, cmap='gray')
                        # plt.show()
                        disease = self.solver.TRAIN_DISEASES[label_idx]
                        print(disease)
                        bbox_ids = superIDtoid[label_idx+1]
                        keys_list = list(current_bbox.keys())
                        overlap = [x for x in bbox_ids if x in keys_list]
                        assert len(overlap) > 0
                        for bbox_id in overlap:
                            out_dir = os.path.join(self.out_dir, disease, 'pos')
                            bbox_mask_orig = current_bbox[bbox_id][:,:, slice_id]
                            bbox_mask = self.resize_bbox_mask(bbox_mask_orig, tagt_size = (320, 320))
                            bbox_name = img_id + '_' +  f"{bbox_id:02}" + '_pos'
                            if count[disease + '_pos'] >= 10:
                                break
                            count[disease + '_pos'] += 1
                            np.save(os.path.join(out_dir, bbox_name + '.npy'), bbox_mask)
                            # Find the indices of non-zero elements in the mask
                            nonzero_indices = np.where(bbox_mask != 0)
                            if len(nonzero_indices[0]) != 0:
                                # Calculate the bounding box coordinates
                                min_row, min_col = np.min(nonzero_indices, axis=1)
                                max_row, max_col = np.max(nonzero_indices, axis=1)
                                # The bounding box is defined by (min_row, min_col) and (max_row, max_col)
                                bounding_box = [min_row, min_col, max_row, max_col]
                                draw_BBox(img, bounding_box, 'input_'+ bbox_name, out_dir)
                                draw_BBox(attr, bounding_box, 'attri_' + bbox_name, out_dir)










def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='bcos_resnet', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='skmtea', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    resnet_model_path_dict = {
        "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-09-22 16:20:52-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
    }

    bcos_resnet_model_path_dict = {
        "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-22 16:24:51-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
    }

    attrinet_model_path_dict = {
        # "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-21 21:19:47--skmtea--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01"
        "skmtea": '/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-22 13:19:07--skmtea--bs=4--lg_ds=32--l_cri=1.0--l1=500.0--l2=1000.0--l3=500.0--l_ctr=0.01'
    }
    if opts.exp_name == 'resnet':
        opts.model_path = resnet_model_path_dict[opts.dataset]
        print("evaluate resnet model: " + opts.model_path)
    if opts.exp_name == 'bcos_resnet':
        opts.model_path = bcos_resnet_model_path_dict[opts.dataset]
        print("evaluate resnet model: " + opts.model_path)
    if opts.exp_name == 'attri-net':
        opts.model_path = attrinet_model_path_dict[opts.dataset]
        print("evaluate attri-net model: " + opts.model_path)
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
    analyser.run()
    # analyser.run_withBB()
    # analyser.plot_centers()


if __name__ == "__main__":

    params = get_arguments()
    main(params)