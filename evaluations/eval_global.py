import numpy as np
from PIL import Image
from train_utils import to_numpy
import os
from models.attrinet_modules import CenterLoss
from models.lgs_classifier import LogisticRegressionModel
import torch
import matplotlib.pyplot as plt


TRAIN_DISEASES= ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
img_size = 320
logreg_dsratio=32


def calculate_Cardio_cs(loss_module, confounder_file_path, outdir):
    confounder = np.loadtxt(confounder_file_path)
    pos_center = loss_module.centers[1].data
    pos_center = to_numpy(pos_center).reshape((320, 320))
    score = sstt(pos_center, confounder, attr_method= "attrinet", ratio=0.1, plot_itsect=True, out_dir=outdir, prefix="Cardio_cs.png")
    print("score: ", score)


def sstt(spu_attr, confounder, attr_method, ratio, plot_itsect, out_dir, prefix):
    # 'lime', 'GCam', 'GB', 'shap', 'gifsplanation','attrinet'
    # get confounder pixel positions
    confounder_pos = np.where(confounder == 1)
    confounder_pos_x = confounder_pos[0]
    confounder_pos_y = confounder_pos[1]
    num_founder_pixels = len(confounder_pos_y)
    confounder_pixels = list(zip(confounder_pos_x, confounder_pos_y))
    confounder_set = set(confounder_pixels)

    # select top 10% pixel with highest value
    all_attr_pixel = 320 * 320
    num_pixels = int(ratio * all_attr_pixel)

    if attr_method == 'attrinet' or attr_method == 'gifsplanation':
        pixel_importance = np.absolute(to_numpy(spu_attr.squeeze()))
    else:
        pixel_importance = spu_attr

    idcs = np.argsort(pixel_importance.flatten())  # from smallest to biggest
    idcs = idcs[::-1]  # if we want the order biggest to smallest, we reverse the indices array
    idcs = idcs[:num_pixels]
    # Compute the corresponding masks for deleting pixels in the given order
    positions = np.array(
        np.unravel_index(idcs, pixel_importance.shape)).T  # first colum, h index, second column, w index
    attri_pos_x = positions[:, 0]
    attri_pos_y = positions[:, 1]
    top_attri_pixels = list(zip(attri_pos_x, attri_pos_y))
    top_attri_set = set(top_attri_pixels)
    inter_set = confounder_set.intersection(top_attri_set)
    hitts = len(inter_set) / num_founder_pixels
    if hitts != 0 and plot_itsect == True:
        pixels = [list(item) for item in inter_set]
        pixels = np.asarray(pixels)
        background = np.zeros((320, 320))
        background[pixels[:, 0], pixels[:, 1]] = 255
        img = Image.fromarray(background)
        img = img.convert("L")
        out_path = os.path.join(out_dir, prefix + "_" + str(hitts) + "_hitts.png")
        img.save(out_path)
    return hitts

def save_img(img, prefix, dir):
    path = os.path.join(dir, prefix)
    plt.imsave(path, img, cmap='gray')

def reshape_weights(weights_1d):
    len = max(weights_1d.shape)
    len_sqrt = int(np.sqrt(len))
    weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
    return weights_2d

def upsample_weights(weights_2d, target_size):
    print(weights_2d.shape)
    source_size = weights_2d.shape[0]
    up_ratio = int(target_size/source_size)
    upsampled_weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))

    return upsampled_weights


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


def load_centers(model_dir):
    suffix = "_best.pth"
    center_losses = {}
    for disease in TRAIN_DISEASES:
        loss = CenterLoss(num_classes=2, feat_dim=img_size * img_size, device=None)
        center_losses[disease] = loss
        center_file_name = "center_of_" + disease + suffix
        center_path = os.path.join(model_dir, center_file_name)
        center_losses[disease].load_state_dict(torch.load(center_path))
    return center_losses


def vis_weights(net_lgs, out_dir):
    # visualize the weights of lgs
    for disease in TRAIN_DISEASES:
        lgs = net_lgs[disease]
        weights = -to_numpy(lgs.linear.weight.data)
        weights = reshape_weights(weights)
        weights = upsample_weights(weights, target_size=320)
        save_img(weights, disease + '_flip_lgs_weights.png', dir=out_dir)




def vis_classcenter(center_losses, out_dir):
    for disease in TRAIN_DISEASES:
        loss_module = center_losses[disease]
        neg_center = loss_module.centers[0].data
        pos_center = loss_module.centers[1].data
        neg_center = to_numpy(neg_center).reshape((320, 320))
        pos_center = to_numpy(pos_center).reshape((320, 320))

        vmax = np.abs(np.asarray([pos_center,neg_center])).flatten().max()
        #vmin = np.abs(np.asarray([pos_center,neg_center])).flatten().min()

        filename = disease + "_flip_pos_centers.png"
        out_path = os.path.join(out_dir, filename)
        plt.imsave(out_path, -pos_center, cmap='gray', vmax=vmax, vmin=-vmax)

        filename = disease + "_flip_neg_centers.png"
        out_path = os.path.join(out_dir, filename)
        plt.imsave(out_path, -neg_center, cmap='gray', vmax=vmax,vmin=-vmax)

        # fig, ax = plt.subplots()
        # # Display the image
        # ax.imshow(-neg_center, cmap='gray')
        # ax.axis('off')
        # filename = disease + "neg_centers.png"
        # out_path = os.path.join(out_dir, filename)
        #
        # fig.savefig(out_path, bbox_inches='tight')
        #
        # fig, ax = plt.subplots()
        # # Display the image
        # ax.imshow(-pos_center, cmap='gray')
        # ax.axis('off')
        # filename = disease + "pos_centers.png"
        # out_path = os.path.join(out_dir, filename)
        #
        # fig.savefig(out_path, bbox_inches='tight')





def main():
    # # model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-02 16:51:03--contam20--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42"
    # model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-23 18:13:03--contam50--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42"
    #
    #
    # model_dir = model_path + "/ckpt"
    # #confounder_file_path = os.path.join(model_path, "tag.txt")
    # confounder_file_path = "./tag.txt"
    # net_lgs = load_lgs(model_dir)
    # out_dir = os.path.join(model_path, "contaim_weights")
    # os.makedirs(out_dir,exist_ok=True)
    # vis_weights(net_lgs, out_dir)
    #
    # center_losses = load_centers(model_dir)
    # vis_classcenter(center_losses, out_dir)
    # calculate_Cardio_cs(center_losses['Cardiomegaly'], confounder_file_path, out_dir)

    # # draw normal model centers and weights
    # model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01"
    # model_dir = model_path + "/ckpt"
    # net_lgs = load_lgs(model_dir)
    # out_dir = os.path.join(model_path, "contaim_weights")
    # os.makedirs(out_dir,exist_ok=True)
    # vis_weights(net_lgs, out_dir)
    #
    # center_losses = load_centers(model_dir)
    # vis_classcenter(center_losses, out_dir)



    # model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-13 11:26:19--contaminated_chexpert--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--guidance_shortcut--l_loc=1500.0--guid_freq=0.0--seed=42"
    model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-23 18:13:03--contam50--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42"
    ## model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2024-01-05 18:56:15--contaminated_chexpert--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--guidance_shortcut--l_loc=1500.0--guid_freq=0.0--seed=42"
    ## model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42"

    model_dir = model_path + "/ckpt"

    confounder_file_path = "./tag.txt"
    net_lgs = load_lgs(model_dir)
    out_dir = os.path.join(model_path, "contaim_weights")
    os.makedirs(out_dir,exist_ok=True)
    vis_weights(net_lgs, out_dir)
    #
    center_losses = load_centers(model_dir)
    vis_classcenter(center_losses, out_dir)
    calculate_Cardio_cs(center_losses['Cardiomegaly'], confounder_file_path, out_dir)

    # # draw normal model centers and weights
    # model_path = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01"
    # model_dir = model_path + "/ckpt"
    # net_lgs = load_lgs(model_dir)
    # out_dir = os.path.join(model_path, "contaim_weights")
    # os.makedirs(out_dir,exist_ok=True)
    # vis_weights(net_lgs, out_dir)
    #
    # center_losses = load_centers(model_dir)
    # vis_classcenter(center_losses, out_dir)








if __name__ == "__main__":

    main()