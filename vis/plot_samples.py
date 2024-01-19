import sys
import os
sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/github_projects/tmi"))

import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from evaluations.model_dict import resnet_models, bcos_resnet_models, attrinet_models, aba_loss_attrinet_models, aba_guidance_attrinet_models, guided_attrinet_models, guided_bcos_resnet_models
import json
import datetime


import numpy as np
from train_utils import to_numpy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import torchvision.transforms as tfs
from data.data_utils import normalize_image, map_image_to_intensity_range
import torch
from models.lgs_classifier import LogisticRegressionModel
from models.attrinet_modules import Discriminator_with_Ada, Generator_with_Ada, CenterLoss
from torch import nn


#
# img_size = 320
# logreg_dsratio=32
# num_class=5
# n_ones=20

def get_lgs_weights(lgs, img_size=320):
    weights_1d = to_numpy(lgs.linear.weight.data)
    len = max(weights_1d.shape)
    len_sqrt = int(np.sqrt(len))
    weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
    source_size = weights_2d.shape[0]
    up_ratio = int(img_size / source_size)
    weights = -np.kron(weights_2d, np.ones((up_ratio, up_ratio))) # flip the sign to be consistent with the -attri_map
    return weights

def compute_weighted_maps(raw_map, lgs):
    raw_map = to_numpy(raw_map).squeeze()
    weights = get_lgs_weights(lgs, raw_map.shape[0])
    weightedmap = np.multiply(raw_map, weights)
    weightedmap = np.where(weightedmap > 0, weightedmap, 0)
    #return -raw_map, -weights, weightedmap
    return weightedmap

def vis_color_map(map, output_dir, prefix):
    attr = to_numpy(map * 0.5 + 0.5).squeeze()
    attri_img = plt.cm.bwr(attr)  # use bwr color map, here negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
    attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
    # attri_img.show()
    attri_img.save(os.path.join(output_dir, prefix + '.jpg'))

def vis_grey_map(map, output_dir, prefix):
    attr = to_numpy(map * 0.5 + 0.5).squeeze()
    # attri_img = plt.cm.bwr(attr)  # use bwr color map, here negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
    attri_img = Image.fromarray((attr * 255).astype(np.uint8))
    attri_img.save(os.path.join(output_dir, prefix + '_grey.jpg'))



# def vis_maps(map, weights, weightedmap, output_dir):
#     vis_color_map(map, output_dir, 'map')
#     vis_color_map(weights, output_dir, 'weights')
#     vis_color_map(weightedmap, output_dir, 'weightedmap')


def preprocess_src_img(src_img_path):
    transforms = tfs.Compose([tfs.Resize((320, 320)), tfs.ToTensor()])
    img = Image.open(src_img_path).convert("L")
    if transforms is not None:
        img = transforms(img)  # return image in range (0,1)
    img = normalize_image(img)
    img = map_image_to_intensity_range(img, -1, 1, percentiles=5)
    img = torch.from_numpy(img)
    return img


def load_lgs(model_dir, TRAIN_DISEASES, img_size, logreg_dsratio):
    # initialize one classifier for one disease
    suffix = "_best.pth"
    net_lgs = {}
    for disease in TRAIN_DISEASES:
        m = LogisticRegressionModel(
            input_size=img_size, num_classes=1, downsample_ratio=logreg_dsratio)
        net_lgs[disease] = m
        c_file_name = "classifier_of_" + disease + suffix
        c_path = os.path.join(model_dir, c_file_name)
        net_lgs[disease].load_state_dict(torch.load(c_path))
    return net_lgs

def load_g(model_dir, num_class, n_ones, img_size):
    net_g = Generator_with_Ada(num_classes=num_class, img_size=img_size,
                                    act_func="relu", n_fc=8, dim_latent=num_class * n_ones,
                                    conv_dim=64, in_channels=1, out_channels=1, repeat_num=6)
    net_g_path = os.path.join(model_dir, 'net_g' + '_best.pth')
    net_g.load_state_dict(torch.load(net_g_path))
    return net_g

def create_task_code(TRAIN_DISEASES, num_class, n_ones):
    """
    Create code for each task, e.g. for task 0, the code is [1, 1, ..., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    latent_z_task = {}
    for i, task in enumerate(TRAIN_DISEASES):
        z = torch.zeros([1, num_class * n_ones], dtype=torch.float)
        start = n_ones * i
        end = n_ones * (i + 1)
        z[:, start:end] = 1
        latent_z_task[task] = z
    return latent_z_task



def prepare_model(model_dir, TRAIN_DISEASES, num_class=5, n_ones=20, img_size=320, logreg_dsratio=32):
    net_lgs = load_lgs(model_dir, TRAIN_DISEASES, img_size, logreg_dsratio)
    net_g = load_g(model_dir, num_class, n_ones, img_size)
    task_codes = create_task_code(TRAIN_DISEASES, num_class, n_ones)
    return net_g, net_lgs, task_codes


def vis_mask_on_src(attr, src, output_dir, prefix):
    attr = to_numpy(attr).squeeze()
    src_img = to_numpy(src * 0.5 + 0.5).squeeze()
    src_img = Image.fromarray(src_img * 255).convert('RGB')
    rgb_attr = np.zeros((attr.shape[0], attr.shape[1], 3), dtype=np.uint8)
    rgb_attr[:, :, 0] = attr * 255
    attr_img = Image.fromarray(rgb_attr.astype(np.uint8)).convert('RGB')
    attr_img.putalpha(200)
    src_img.paste(attr_img, (0, 0), attr_img)
    src_img.save(os.path.join(output_dir, prefix + '_attr_on_src.jpg'))









def vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, src_img_path, output_dir, color_map, img_name):
    img = preprocess_src_img(src_img_path)
    vis_grey_map(img, output_dir, img_name+"src_img")
    raw_attri_maps = []
    prefix_lists= []
    dests=[]
    for tgt_label in TRAIN_DISEASES:
        dest, attr = net_g(img, task_codes[tgt_label])
        raw_attri_maps.append(to_numpy(-attr).squeeze())
        pred_logits = net_lgs[tgt_label](attr)
        prob = torch.sigmoid(pred_logits)
        prefix_lists.append('_raw_map' + "_prob:" + str(prob.item()))
        dests.append(dest)
    #
    max_abs_value = np.abs(np.array(raw_attri_maps)).max()
    for i in range(len(raw_attri_maps)):
        map = raw_attri_maps[i]
        scaled_attri_map = (map / max_abs_value)
        if color_map=="color":
            vis_color_map(scaled_attri_map, output_dir, img_name + TRAIN_DISEASES[i] + '_raw_map'+ prefix_lists[i])

        if color_map == "grey":
            vis_grey_map(scaled_attri_map, output_dir, img_name + TRAIN_DISEASES[i] + "grey_" + prefix_lists[i])
            vis_grey_map(dests[i], output_dir, img_name + TRAIN_DISEASES[i] + "grey_dest_" + prefix_lists[i])
        if color_map == "weighted":
            weightedmap = compute_weighted_maps(map, net_lgs[TRAIN_DISEASES[i]])
            vis_color_map(weightedmap, output_dir,
                          img_name + TRAIN_DISEASES[i] + '_weighted_map' +prefix_lists[i])
            #vis_mask_on_src(weightedmap, img, output_dir, prefix=img_name + TRAIN_DISEASES[i])




def vis_one_downsampled_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, src_img_path, output_dir):
    down = nn.AvgPool2d(32, stride=32)
    img = preprocess_src_img(src_img_path)
    raw_down_attri_maps = []
    prefix_lists = []
    for tgt_label in TRAIN_DISEASES:
        dest, attr = net_g(img, task_codes[tgt_label])
        pred_logits = net_lgs[tgt_label](attr)
        prob = torch.sigmoid(pred_logits)
        down_sampled_attr = np.kron(to_numpy(down(-attr)), np.ones((32, 32)))
        raw_down_attri_maps.append(down_sampled_attr.squeeze())
        prefix_lists.append('_downsampled_raw_map'+ "_prob:" + str(prob.item()))
    #
    max_abs_value = np.abs(np.array(raw_down_attri_maps)).max()
    for i in range(len(raw_down_attri_maps)):
        map = raw_down_attri_maps[i]
        scaled_attri_map = (map / max_abs_value)
        vis_color_map(scaled_attri_map, output_dir, TRAIN_DISEASES[i] + prefix_lists[i])
        vis_grey_map(scaled_attri_map, output_dir, TRAIN_DISEASES[i] + prefix_lists[i])


def vis_lgs_weights(net_lgs, output_dir,color_map):
    for disease in net_lgs.keys():
        weights = get_lgs_weights(net_lgs[disease])
        if color_map == 'color':
            vis_color_map(weights, output_dir, disease + '_weights_color')
        if color_map == 'grey':
            plt.imsave(os.path.join(output_dir, disease + '_weights' + '_grey.jpg'), weights, cmap='gray')


            # attri_img = Image.fromarray((weights * 255).astype(np.uint8))
            # attri_img.save(os.path.join(output_dir, disease + '_weights' + '_grey.jpg'))




def load_thresholds(attrinet_model_path, TRAIN_DISEASES):
    threshold_path = os.path.join(attrinet_model_path + "/class_sensitivity_result_dir", "best_threshold.txt")
    threshold = np.loadtxt(open(threshold_path))
    best_threshold = {}
    for c in range(len(TRAIN_DISEASES)):
        disease = TRAIN_DISEASES[c]
        best_threshold[disease] = threshold[c]
    return best_threshold



# class AttriNet_visualizer():
#     """
#     Analyse the classification performance of the model.
#     """
#     def __init__(self, model_path, TRAIN_DISEASES):
#         self.model_path = model_path
#         self.TRAIN_DISEASES = TRAIN_DISEASES
#         self.model_dir = os.path.join(model_path, "/ckpt")
#         self.net_g, self.net_lgs, self.task_codes = self.prepare_model(self.model_dir, self.TRAIN_DISEASES)
#
#     def load_thresholds(self, TRAIN_DISEASES):
#         threshold_path = os.path.join(self.model_path + "/class_sensitivity_result_dir", "best_threshold.txt")
#         threshold = np.loadtxt(open(threshold_path))
#         best_threshold = {}
#         for c in range(len(TRAIN_DISEASES)):
#             disease = TRAIN_DISEASES[c]
#             best_threshold[disease] = threshold[c]
#         return best_threshold
#
#     def prepare_model(self, TRAIN_DISEASES, num_class=5, n_ones=20, img_size=320, logreg_dsratio=32):
#         net_lgs = load_lgs(self.model_dir, TRAIN_DISEASES, img_size, logreg_dsratio)
#         net_g = load_g(self.model_dir, num_class, n_ones, img_size)
#         task_codes = create_task_code(TRAIN_DISEASES, num_class, n_ones)
#         return net_g, net_lgs, task_codes
#
#
#


if __name__ == "__main__":

    '''
    attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    model_dir = attrinet_model_path + "/ckpt"

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]



    # Fig1
    
    attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    model_dir = attrinet_model_path + "/ckpt"

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    
    # src_image_name = "patient64749_study1_view1_frontal" # for weighted map plot
    #src_image_name = "patient64763_study1_view1_frontal" # for new framework plot
    src_image_name = "patient65028_study1_view1_frontal"  # for new framework plot
    src_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test", src_image_name.split("_")[0], src_image_name.split("_")[1], src_image_name.split("_")[2]+"_"+src_image_name.split("_")[3]+".jpg")

    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results/fig1_samples"
    out_dir = os.path.join(out_dir,src_image_name)
    os.makedirs(out_dir, exist_ok=True)

    net_g, net_lgs, task_codes = prepare_model(model_dir, TRAIN_DISEASES=TRAIN_DISEASES, num_class=5, n_ones=20,
                                               img_size=320, logreg_dsratio=32)

    thresholds = load_thresholds(attrinet_model_path, TRAIN_DISEASES)

    #get_map_and_lgs(model_dir, src_img_path, tgt_label="Cardiomegaly", output_dir=out_dir)

    vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, src_img_path, output_dir=out_dir,color_map='color',img_name=src_image_name)
    vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, src_img_path, output_dir=out_dir,color_map='weighted',img_name=src_image_name)
    vis_lgs_weights(net_lgs, output_dir=out_dir, color_map="color")
    vis_one_downsampled_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, src_img_path, output_dir=out_dir)


    
    '''




    # Fig2
    """
    
    attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    model_dir = attrinet_model_path + "/ckpt"

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    
    
    positive_sample_cardiomegaly = "patient65205_study1_view1_frontal"  # for new framework plot
    positive_sample_cardiomegaly_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test",
                                positive_sample_cardiomegaly.split("_")[0], positive_sample_cardiomegaly.split("_")[1],
                                positive_sample_cardiomegaly.split("_")[2] + "_" + positive_sample_cardiomegaly.split("_")[3] + ".jpg")

    negative_sample_cardiomegaly = "patient65013_study1_view1_frontal"  # for new framework plot
    negative_sample_cardiomegaly_img_path = os.path.join(
        "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test",
        negative_sample_cardiomegaly.split("_")[0], negative_sample_cardiomegaly.split("_")[1],
        negative_sample_cardiomegaly.split("_")[2] + "_" + negative_sample_cardiomegaly.split("_")[3] + ".jpg")

    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results/fig2_samples"
    os.makedirs(out_dir, exist_ok=True)

    net_g, net_lgs, task_codes = prepare_model(model_dir, TRAIN_DISEASES=TRAIN_DISEASES, num_class=5, n_ones=20,
                                               img_size=320, logreg_dsratio=32)


    vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, positive_sample_cardiomegaly_img_path, output_dir=out_dir, color_map="grey", img_name=positive_sample_cardiomegaly)
    vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, negative_sample_cardiomegaly_img_path, output_dir=out_dir,color_map="grey", img_name=negative_sample_cardiomegaly)

    
    """


    
    # Fig9
    
    attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    model_dir = attrinet_model_path + "/ckpt"

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    
    
    sample_name = "patient64763_study1_view1_frontal"  # for all local explanations plots
    img_path = os.path.join(
        "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test",
        sample_name.split("_")[0], sample_name.split("_")[1],
        sample_name.split("_")[2] + "_" + sample_name.split("_")[3] + ".jpg")

    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results/fig9_samples"
    os.makedirs(out_dir, exist_ok=True)

    net_g, net_lgs, task_codes = prepare_model(model_dir, TRAIN_DISEASES=TRAIN_DISEASES, num_class=5, n_ones=20,
                                               img_size=320, logreg_dsratio=32)

    vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, img_path,
                   output_dir=out_dir, color_map="weighted", img_name=sample_name)
    




    # supplementary all conterfactuals, use nih model, more samples to select from

    #
    # # use model trained on NIH
    # attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-11 15:40:33--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    # model_dir = attrinet_model_path + "/ckpt"
    #
    # TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
    #
    #
    # samples_nih={
    #     "positive_sample_Atelectasis": "00020393_003.png",# for new framework plot
    #     "negative_sample_Atelectasis": "00029102_003.png",  # for new framework plot
    #
    #     "positive_sample_Cardiomegaly": "00004533_018.png",  # for new framework plot
    #     "negative_sample_Cardiomegaly": "00007728_007.png",  # for new framework plot
    #
    #     "positive_sample_Consolidation": "00021201_022.png",  # for new framework plot
    #     "negative_sample_Consolidation": "00003148_008.png",  # for new framework plot
    #
    #     "positive_sample_Edema": "00019625_002.png",  # for new framework plot
    #     "negative_sample_Edema": "00023168_009.png",  # for new framework plot
    #
    #     "positive_sample_Effusion": "00022369_011.png",  # for new framework plot
    #     "negative_sample_Effusion": "00026287_007.png",  # for new framework plot
    # }
    #

    """

    # use model trained on CheXpert, supplementray material fig A1 counterfactuals all
    
    attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    model_dir = attrinet_model_path + "/ckpt"

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

    samples_chexpert={
        "positive_sample_Atelectasis": "patient65044_study1_view1_frontal",# for new framework plot
        "negative_sample_Atelectasis": "patient64957_study1_view1_frontal",  # for new framework plot

        "positive_sample_Cardiomegaly": "patient64862_study1_view1_frontal",  # for new framework plot
        "negative_sample_Cardiomegaly": "patient64860_study1_view1_frontal",  # for new framework plot

        "positive_sample_Consolidation": "patient65053_study1_view1_frontal",  # for new framework plot
        "negative_sample_Consolidation": "patient64782_study1_view1_frontal",  # for new framework plot

        "positive_sample_Edema": "patient65188_study1_view1_frontal",  # for new framework plot
        "negative_sample_Edema": "patient64789_study1_view1_frontal",  # for new framework plot

        "positive_sample_Pleural Effusion": "patient65070_study1_view1_frontal",  # for new framework plot
        "negative_sample_Pleural Effusion": "patient64766_study1_view1_frontal",  # for new framework plot
    }
    samples = samples_chexpert

    for disease in TRAIN_DISEASES:
        postive_sample_name = samples["positive_sample_" + disease]
        negative_sample_name = samples["negative_sample_" + disease]

        # positive_sample_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled", postive_sample_name)
        # negative_sample_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled", negative_sample_name)

        positive_sample_img_path = os.path.join(
            "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test",
            postive_sample_name.split("_")[0], postive_sample_name.split("_")[1],
            postive_sample_name.split("_")[2] + "_" + postive_sample_name.split("_")[3] + ".jpg")


        negative_sample_img_path = os.path.join(
            "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test",
            negative_sample_name.split("_")[0], negative_sample_name.split("_")[1],
            negative_sample_name.split("_")[2] + "_" + negative_sample_name.split("_")[3] + ".jpg")


        out_dir = os.path.join("/mnt/qb/work/baumgartner/sun22/TMI_exps/results/supple_fig2_samples", disease)
        os.makedirs(out_dir, exist_ok=True)

        net_g, net_lgs, task_codes = prepare_model(model_dir, TRAIN_DISEASES=TRAIN_DISEASES, num_class=5, n_ones=20,
                                                   img_size=320, logreg_dsratio=32)

        vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, positive_sample_img_path,
                       output_dir=out_dir, color_map="grey", img_name=postive_sample_name)
        vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, negative_sample_img_path,
                       output_dir=out_dir, color_map="grey", img_name=negative_sample_name)
    """







    """

    # supplementary material ablation study of loss terms.
    # use model trained on NIH

    aba_loss_attrinet_models = {
        "cls": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 18:11:38--nih_chestxray--l_cri=0.0--l1=0.0--l2=0.0--l_cls=100.0--l_ctr=0.0--no_guidance--seed=42",
        "cls_adv": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 18:14:42--nih_chestxray--l_cri=1.0--l1=0.0--l2=0.0--l_cls=100.0--l_ctr=0.0--no_guidance--seed=42",
        "cls_adv_reg": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 18:14:46--nih_chestxray--l_cri=1.0--l1=100.0--l2=200.0--l_cls=100.0--l_ctr=0.0--no_guidance--seed=42",
        "cls_adv_reg_ctr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 17:29:36--nih_chestxray--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
        "all": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-11 15:40:33--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    }

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

    samples_nih={
        "positive_sample_Cardiomegaly": "00000211_019.png",  # for new framework plot
        "negative_sample_Cardiomegaly": "00030194_000.png",  # for new framework plot
    }
    positive_sample_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled", samples_nih["positive_sample_Cardiomegaly"])
    negative_sample_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled", samples_nih["negative_sample_Cardiomegaly"])


    for model_name, model_path in aba_loss_attrinet_models.items():
        model_dir = model_path + "/ckpt"

        out_dir = os.path.join("/mnt/qb/work/baumgartner/sun22/TMI_exps/results/supple_figB_samples", model_name)
        os.makedirs(out_dir, exist_ok=True)

        net_g, net_lgs, task_codes = prepare_model(model_dir, TRAIN_DISEASES=TRAIN_DISEASES, num_class=5, n_ones=20,
                                                   img_size=320, logreg_dsratio=32)

        vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, positive_sample_img_path,
                       output_dir=out_dir, color_map="grey", img_name=samples_nih["positive_sample_Cardiomegaly"])
        vis_one_sample(net_g, net_lgs, task_codes, TRAIN_DISEASES, negative_sample_img_path,
                       output_dir=out_dir, color_map="grey", img_name=samples_nih["negative_sample_Cardiomegaly"])
    """