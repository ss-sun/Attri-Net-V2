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



TRAIN_DISEASES= ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
img_size = 320
logreg_dsratio=32
num_class=5
n_ones=20



def compute_maps(map, lgs):
    map = to_numpy(map).squeeze()
    weights_1d = to_numpy(lgs.linear.weight.data)
    len = max(weights_1d.shape)
    len_sqrt = int(np.sqrt(len))
    weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
    source_size = weights_2d.shape[0]
    up_ratio = int(map.shape[0] / source_size)
    weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))
    weightedmap = np.multiply(map, weights)
    return -map, -weights, weightedmap

def vis_color_map(map, output_dir, prefix):
    if prefix == "weightedmap":
        map = np.where(map > 0, map, 0)
    attr = to_numpy(map * 0.5 + 0.5).squeeze()
    attri_img = plt.cm.bwr(attr)  # use bwr color map, here negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
    attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
    attri_img.show()
    attri_img.save(os.path.join(output_dir, prefix + '_attri.jpg'))

def vis_maps(map, lgs, output_dir):
    map, weights, weightedmap = compute_maps(map, lgs)
    vis_color_map(map, output_dir, 'map')
    vis_color_map(weights, output_dir, 'weights')
    vis_color_map(weightedmap, output_dir, 'weightedmap')


def preprocess_src_img(src_img_path):
    transforms = tfs.Compose([tfs.Resize((320, 320)), tfs.ToTensor()])
    img = Image.open(src_img_path).convert("L")
    if transforms is not None:
        img = transforms(img)  # return image in range (0,1)
    img = normalize_image(img)
    img = map_image_to_intensity_range(img, -1, 1, percentiles=5)
    img = torch.from_numpy(img)
    return img



def load_lgs(model_dir):
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

def load_g(model_dir):
    net_g = Generator_with_Ada(num_classes=num_class, img_size=img_size,
                                    act_func="relu", n_fc=8, dim_latent=num_class * n_ones,
                                    conv_dim=64, in_channels=1, out_channels=1, repeat_num=6)
    net_g_path = os.path.join(model_dir, 'net_g' + '_best.pth')
    net_g.load_state_dict(torch.load(net_g_path))
    return net_g


def create_task_code():
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


def get_map_and_lgs(model_dir, src_img_path, tgt_label, output_dir):
    img = preprocess_src_img(src_img_path)
    net_lgs = load_lgs(model_dir)
    net_g = load_g(model_dir)
    task_codes = create_task_code()
    dest, attr = net_g(img, task_codes[tgt_label])
    vis_maps(attr, net_lgs[tgt_label], output_dir)




if __name__ == "__main__":
    attrinet_model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"
    model_dir = attrinet_model_path + "/ckpt"

    src_image_name = "patient64749_study1_view1_frontal"
    src_img_path = os.path.join("/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled/test", src_image_name.split("_")[0], src_image_name.split("_")[1], src_image_name.split("_")[2]+"_"+src_image_name.split("_")[3]+".jpg")
    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results/plots"

    get_map_and_lgs(model_dir, src_img_path, tgt_label="Cardiomegaly", output_dir=out_dir)

