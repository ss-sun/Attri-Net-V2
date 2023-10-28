import os
import numpy as np
import torch
import random
import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from train_utils import to_numpy
from tqdm import tqdm
import pandas as pd
import json
from pycocotools import mask
import cv2
from PIL import Image, ImageDraw

LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                       "Cardiomegaly",
                       "Lung Lesion",
                       "Airspace Opacity",
                       "Edema",
                       "Consolidation",
                       "Atelectasis",
                       "Pneumothorax",
                       "Pleural Effusion",
                       "Support Devices"]

TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],

def str2bool(v):
    return v.lower() in ('true')

def reshape_weights(weights_1d):
    len = max(weights_1d.shape)
    len_sqrt = int(np.sqrt(len))
    weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
    return weights_2d

def upsample_weights(weights_2d, target_size):
    source_size = weights_2d.shape[0]
    up_ratio = int(target_size/source_size)
    upsampled_weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))
    return upsampled_weights




class pixel_sensitivity_analyser():
    def __init__(self, solver, config):
        gt_seg_dir = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"

        self.solver = solver
        self.result_dir = config.result_dir
        self.train_diseases = self.solver.TRAIN_DISEASES
        self.attr_method = config.attr_method
        os.makedirs(self.result_dir, exist_ok=True)
        self.attr_dir = os.path.join(self.result_dir, self.attr_method)
        os.makedirs(self.attr_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plot = True

        print('compute pixel sensitivity on dataset ' + config.dataset)
        print('use explainer: ' + self.attr_method)

        with open(gt_seg_dir) as json_file:
            self.gt_dict = json.load(json_file)
        # self.test_df = pd.read_csv(src_csv)
        self.test_df = self.solver.test_loader.dataset.df

    def run(self):
        self.compute_pixel_sensitivity()
        self.calculate_statics("pixel_sensitivity")

    def calculate_statics(self, which_statics):
        sum = 0
        if which_statics == "IoU":
            df = pd.read_csv(os.path.join(self.attr_dir, 'IoU_results.csv'), index_col=0)
        if which_statics == "pixel_sensitivity":
            df = pd.read_csv(os.path.join(self.attr_dir, 'pixel_sensitivity.csv'), index_col=0)

        for disease in self.train_diseases:
            values = df[disease].tolist()
            values = np.asarray(values)
            mean = np.nanmean(values)
            print(which_statics + " of " + disease + " is: ", mean)
            sum += mean
        mean_over_disease = sum / len(self.train_diseases)
        print(which_statics + " over all diseases is: ", mean_over_disease)





    def compute_pixel_sensitivity(self):
        results = {}
        IoU_results = {}
        for index, row in self.test_df.iterrows():
            print("idx:", index)
            if "lateral" in row['Path']:
                continue
            # get data
            data = self.solver.test_loader.dataset[int(index)]
            test_data, test_labels = data["img"], data["label"]
            test_data = torch.from_numpy(test_data[None])
            test_labels = torch.from_numpy(test_labels[None])
            test_data = test_data.to(self.device)
            # test_labels = test_labels.to(self.device)
            path = row['Path']
            cxr_id = self.get_cxr_id(path)
            for disease in self.train_diseases:

                if cxr_id in results:
                    if disease in results[cxr_id]:
                        print(f'Check for duplicates for {disease} for {cxr_id}')
                        break
                    else:
                        results[cxr_id][disease] = 0
                else:
                    # get ground truth binary mask
                    if cxr_id not in self.gt_dict:
                        continue
                    else:
                        results[cxr_id] = {}
                        results[cxr_id][disease] = 0
                        IoU_results[cxr_id] = {}
                        IoU_results[cxr_id][disease] = 0

                gt_mask = self.get_gt_mask(cxr_id, disease)
                if self.attr_method == 'attri-net':
                    task_code = self.solver.latent_z_task[disease].to(self.device)
                    _, attri = self.solver.net_g(test_data, task_code)
                    lgs = self.solver.net_lgs[disease]
                    attri = self.get_weightedmap(attri, lgs)
                    hit, pos = self.get_hit(attri, gt_mask, weighted=True)


                    # hit, pos = self.get_hit_top100(attri, gt_mask, weighted=True)
                    # hit, pos = self.get_hit(attri, gt_mask)
                    # hit, pos = self.get_hit_top100(attri, gt_mask)

                    results[cxr_id][disease] = hit
                    attri_mask = self.attri_to_segmentation(attri)
                    if self.plot:
                        if hit == 0:
                            self.plot_hit(attri, attri_mask, gt_mask, test_data, pos, prefix=f'{cxr_id}_{disease}_miss')
                        if hit == 1:
                            self.plot_hit(attri, attri_mask, gt_mask, test_data, pos, prefix=f'{cxr_id}_{disease}_hit')


                    iou_score = self.calculate_iou(attri_mask, gt_mask)
                    IoU_results[cxr_id][disease] = iou_score
                else:
                    label_idx=self.solver.TRAIN_DISEASES.index(disease)
                    attri = self.solver.get_attributes(test_data, label_idx, positive_only=True)
                    attri = to_numpy(attri).squeeze()
                    hit, pos = self.get_hit(attri, gt_mask, weighted=True)

                    #hit, pos = self.get_hit_top100(attri, gt_mask, weighted=True)
                    # hit, pos = self.get_hit(attri, gt_mask)
                    # hit, pos = self.get_hit_top100(attri, gt_mask)

                    results[cxr_id][disease] = hit
                    attri_mask = self.attri_to_segmentation(attri)
                    if self.plot:
                        if hit == 0:
                            self.plot_hit(attri, attri_mask, gt_mask, test_data, pos, prefix=f'{cxr_id}_{disease}_miss')
                        if hit == 1:
                            self.plot_hit(attri, attri_mask, gt_mask, test_data, pos, prefix=f'{cxr_id}_{disease}_hit')

                    iou_score = self.calculate_iou(attri_mask, gt_mask)
                    IoU_results[cxr_id][disease] = iou_score





        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(os.path.join(self.attr_dir, 'pixel_sensitivity.csv'))
        IoU_results_df = pd.DataFrame.from_dict(IoU_results, orient='index')
        IoU_results_df.to_csv(os.path.join(self.attr_dir, 'IoU_results.csv'))
        return results_df

    def plot_hit(self, attri, attri_mask, gt_mask, src_img, pos, prefix):
        src_img = to_numpy(src_img * 0.5 + 0.5).squeeze()
        src_img = Image.fromarray(src_img * 255).convert('RGB')
        rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = gt_mask * 255
        rgb_mask[:, :, 2] = attri_mask * 255
        # rgb_mask[pos[0], pos[1], 0] = 255
        img = Image.fromarray(rgb_mask).convert('RGB')
        # Create a new ImageDraw object
        draw = ImageDraw.Draw(img)
        # Set the size of the dot
        size = 5
        # Calculate the coordinates of the circle
        x0 = pos[1] - size
        y0 = pos[0] - size
        x1 = pos[1] + size
        y1 = pos[0] + size
        img.putalpha(50)
        # Draw the circle
        draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
        src_img.paste(img, (0, 0), img)
        # src_img.show()
        src_img.save(os.path.join(self.attr_dir, prefix +'.jpg'))

        flip_mask = -0.5 * to_numpy(attri).squeeze() + 0.5
        flip_mask = Image.fromarray(flip_mask * 255).convert('RGB')
        draw = ImageDraw.Draw(flip_mask)
        draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
        # flip_mask.show()
        flip_mask.save(os.path.join(self.attr_dir, prefix + '_mask.jpg'))

    def calculate_iou(self, attri_mask, gt_mask):
        intersection = np.logical_and(attri_mask, gt_mask)
        union = np.logical_or(attri_mask, gt_mask)

        if np.sum(union) == 0:
            iou_score = np.nan
        elif np.sum(attri_mask) == 0 or np.sum(gt_mask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
        return iou_score

    def get_weightedmap(self, mask, lgs):
        weights = to_numpy(lgs.linear.weight.data)
        weights = reshape_weights(weights)
        weights = upsample_weights(weights, target_size=320)
        weightedmap = np.multiply(to_numpy(mask.squeeze()), weights)

        return weightedmap




    def get_hit(self, mask, gt_mask, weighted=False):

        # if self.attr_method == 'attri-net':
        #     mask = np.absolute(to_numpy(mask.squeeze()))
        if weighted:
            mask = mask
        else:
            if self.attr_method == 'attri-net':
                mask = np.absolute(to_numpy(mask.squeeze()))
            else:
                mask = mask

        x = np.unravel_index(np.argmax(mask, axis=None), mask.shape)[0]
        y = np.unravel_index(np.argmax(mask, axis=None), mask.shape)[1]

        assert (gt_mask.shape == mask.shape)
        if np.sum(gt_mask) == 0:
            hit = np.nan
            pos = np.nan
        else:
            if (gt_mask[x][y] == 1):
                hit = 1
                pos = (x, y)
            if (gt_mask[x][y] == 0):
                hit = 0
                pos = (x, y)
        return hit, pos

    def get_hit_top100(self, mask, gt_mask, weighted=False):

        if weighted:
            pixel_importance = mask
        else:
            if self.attr_method == 'attri-net':
                pixel_importance = np.absolute(to_numpy(mask.squeeze()))
            else:
                pixel_importance = mask

        assert (gt_mask.shape == pixel_importance.shape)
        if np.sum(gt_mask) == 0:
            hit = np.nan
            pos = np.nan
        else:
            idcs = np.argsort(pixel_importance.flatten())  # from smallest to biggest
            idcs = idcs[::-1]  # if we want the order biggest to smallest, we reverse the indices array
            idcs = idcs[:50]
            # Compute the corresponding masks for deleting pixels in the given order
            positions = np.array(
                np.unravel_index(idcs, pixel_importance.shape)).T  # first colum, h index, second column, w index
            attri_pos_x = positions[:, 0]
            attri_pos_y = positions[:, 1]
            top_attri_pixels = list(zip(attri_pos_x, attri_pos_y))
            top_attri_set = set(top_attri_pixels)

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
        return hit, pos




    def get_cxr_id(self, path):
        # Remove the "test/" prefix and ".jpg" suffix
        filename = path.replace("test/", "").replace(".jpg", "")
        # Split the filename into its components
        id_components = filename.split("/")
        # Join the components together with underscores
        cxr_id = "_".join(id_components)
        return cxr_id

    def get_gt_mask(self, cxr_id, disease):
        gt_item = self.gt_dict[cxr_id][disease]
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

    def attri_to_segmentation(self, attri_map, threshold=np.nan, smoothing=False, k=0):

        if self.attr_method == 'attri-net':
            # both negative value and positive value are important for segmentation. The larger absolute value indicates larger changes.
            attri_map = np.absolute(to_numpy(attri_map).squeeze())
        if self.attr_method != 'attri-net':
            attri_map = to_numpy(attri_map).squeeze()
            #print("normalization is not implemented for this method")

        # # normalize heatmap
        mask = attri_map - attri_map.min()
        mask = mask / (mask.max())

        # use Otsu's method to find threshold if no threshold is passed in
        if np.isnan(threshold):
            mask = np.uint8(255 * mask)
            if smoothing:
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                gray_img = cv2.boxFilter(cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY),
                                         -1, (k, k))
                mask = 255 - gray_img

            # Apply Otsu's thresholding to create a binary image
            threshold_value, binary_image = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
            # draw out contours
            cnts = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            polygons = []
            for cnt in cnts:
                if len(cnt) > 1:
                    polygons.append([list(pt[0]) for pt in cnt])

            # create segmentation based on contour
            img_dims = (mask.shape[1], mask.shape[0])
            segmentation_output = Image.new('1', img_dims)
            for polygon in polygons:
                coords = [(point[0], point[1]) for point in polygon]
                ImageDraw.Draw(segmentation_output).polygon(coords,
                                                            outline=1,
                                                            fill=1)
            segmentation = np.array(segmentation_output, dtype="int")
        else:
            segmentation = np.array(mask > threshold, dtype="int")
        return segmentation

















def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='attri-net', choices=['resnet', 'attri-net'])
    parser.add_argument('--attr_method', type=str, default='attri-net',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'attri-net' , 'gifsplanation'")

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'nih_chestxray', 'vindr_cxr'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()

    # resnet_model_path_dict = {
    #     # "chexpert": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-17 10:40:08-chexpert-bs=8-lr=0.0001-weight_decay=1e-05",
    #     "chexpert": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-05-02 16:23:22-chexpert--official_datasplit-orientation=all-bs=8-lr=0.0001-weight_decay=1e-05",
    #     "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:22-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
    #     "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:34-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05"
    # }

    # attrinet_model_path_dict = {
    #     # "chexpert": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 17:52:55--chexpert--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    #     "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-05-02 16:23:38--chexpert--official_datasplit-orientation=all--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    #     "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
    #     "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 18:10:38--vindr_cxr--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01"
    # }

    resnet_model_path_dict = {
        "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-07-28 13:27:38-chexpert--official_datasplit-orientation=Frontal-augmentation=previous-bs=4-lr=0.0001-weight_decay=1e-05",
        "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:22-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
        "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:34-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05",
        "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-09-22 16:20:52-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
    }

    bcos_resnet_model_path_dict = {
        "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:18:35-chexpert-bs=8-lr=0.0001-weight_decay=1e-05",
        "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:19:08-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
        "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:17:29-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05",
        "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-22 16:24:51-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
    }


    attrinet_model_path_dict = {
        "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 18:10:38--vindr_cxr--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-22 13:19:07--skmtea--bs=4--lg_ds=32--l_cri=1.0--l1=500.0--l2=1000.0--l3=500.0--l_ctr=0.01",
    }



    if opts.exp_name == 'resnet':
        opts.model_path = resnet_model_path_dict[opts.dataset]
        print("evaluate resnet model: " + opts.model_path)
        opts.result_dir = os.path.join(opts.model_path, "result_dir")

    if opts.exp_name == 'attri-net':
        opts.model_path = attrinet_model_path_dict[opts.dataset]
        print("evaluate attri-net model: " + opts.model_path)
        opts.result_dir = os.path.join(opts.model_path, "result_dir")
        # Configurations of networks
        opts.image_size = 320
        opts.n_fc = 8
        opts.n_ones = 20
        opts.num_out_channels = 1
        opts.lgs_downsample_ratio = 32

    if opts.exp_name == 'bcos_resnet':
        opts.model_path = bcos_resnet_model_path_dict[opts.dataset]
        print("evaluate resnet model: " + opts.model_path)


    return opts


def prep_solver(datamodule, exp_configs):
    data_loader = {}
    if "resnet" in exp_configs.exp_name:
        data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        testBB_loader = datamodule.test_BB_dataloader(batch_size=1, shuffle=False)
        #data_loader = None
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
        testBB_loader = datamodule.test_BB_dataloader(batch_size=1, shuffle=False)
        # data_loader = None
        solver = task_switch_solver(exp_configs, data_loader=data_loader)
    return solver, testBB_loader



def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    solver, testBB_loader = prep_solver(datamodule, config)
    analyser = pixel_sensitivity_analyser(solver, testBB_loader, config)
    analyser.run()





if __name__ == "__main__":
    params = get_arguments()
    main(params)