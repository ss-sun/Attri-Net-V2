import pandas as pd
import os
import numpy as np
import json
from PIL import Image
from PIL import ImageDraw



def draw_speudo_bbox(imgs_dict, pseudo_bbox_dict_path, src_img_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    with open(pseudo_bbox_dict_path) as json_file:
        pseudo_bbox_dict = json.load(json_file)

    for disease in imgs_dict.keys():
        if "CheXpert" in src_img_dir:
            file_name = disease + imgs_dict[disease].split("/")[-3] + "_" + imgs_dict[disease].split("/")[-2] + "_" + imgs_dict[disease].split("/")[-1]
            img = Image.open(src_img_dir + imgs_dict[disease])
            img = img.resize((320, 320))
        if "NIH" in src_img_dir:
            file_name = disease + imgs_dict[disease]
            img = Image.open(os.path.join(src_img_dir, imgs_dict[disease]))
        if "Vindr" in src_img_dir:
            file_name = disease + imgs_dict[disease]
            img = Image.open(os.path.join(src_img_dir, imgs_dict[disease]))

        img.save(os.path.join(dest_dir, "src_" + file_name))
        bbox = pseudo_bbox_dict[disease]
        rgb_img = img.convert('RGB')

        rgb_mask = np.zeros((rgb_img.size[0], rgb_img.size[1], 3), dtype=np.uint8)
        mask_img = Image.fromarray(rgb_mask).convert('RGB')

        draw = ImageDraw.Draw(mask_img)
        outline_color = (0, 255, 0)
        outline_width = 5

        draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline=outline_color,
                       width=outline_width)
        del draw

        mask_img.putalpha(75)
        rgb_img.paste(mask_img, (0, 0), mask_img)
        rgb_img.save(os.path.join(dest_dir, "with_pseudo_bbox_" + file_name))





def draw_speudo_mask(imgs_dict, pseudo_mask_dict_path, src_img_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    with open(pseudo_mask_dict_path) as json_file:
        pseudo_mask_dict = json.load(json_file)

    for disease in imgs_dict.keys():
        if "CheXpert" in src_img_dir:
            file_name = disease + imgs_dict[disease].split("/")[-3] + "_" + imgs_dict[disease].split("/")[-2] + "_" + imgs_dict[disease].split("/")[-1]
            img = Image.open(src_img_dir + imgs_dict[disease])
            img = img.resize((320, 320))
        if "NIH" in src_img_dir:
            file_name = disease + imgs_dict[disease]
            img = Image.open(os.path.join(src_img_dir + imgs_dict[disease]))
        if "Vindr" in src_img_dir:
            file_name = disease + imgs_dict[disease]
            img = Image.open(os.path.join(src_img_dir, imgs_dict[disease]))

        img.save(os.path.join(dest_dir, "src_" + file_name))

        rgb_img = img.convert('RGB')
        mask = np.array(pseudo_mask_dict[disease])

        rgb_mask = np.zeros((rgb_img.size[0], rgb_img.size[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = mask * 255
        mask_img = Image.fromarray(rgb_mask).convert('RGB')

        mask_img.putalpha(50)
        rgb_img.paste(mask_img, (0, 0), mask_img)
        rgb_img.save(os.path.join(dest_dir, "with_pseudo_mask_" + file_name))



def draw_weighted_speudo_mask(imgs_dict, weighted_pseudo_mask_dict_path, src_img_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    with open(weighted_pseudo_mask_dict_path) as json_file:
        weighted_pseudo_mask_dict = json.load(json_file)

    for disease in imgs_dict.keys():
        if "CheXpert" in src_img_dir:
            file_name = disease + imgs_dict[disease].split("/")[-3] + "_" + imgs_dict[disease].split("/")[-2] + "_" + imgs_dict[disease].split("/")[-1]
            img = Image.open(src_img_dir + imgs_dict[disease])
            img = img.resize((320, 320))
        if "NIH" in src_img_dir:
            file_name = disease + imgs_dict[disease]
            img = Image.open(os.path.join(src_img_dir + imgs_dict[disease]))
        if "Vindr" in src_img_dir:
            file_name = disease + imgs_dict[disease]
            img = Image.open(os.path.join(src_img_dir, imgs_dict[disease]))

        img.save(os.path.join(dest_dir, "src_" + file_name))

        rgb_img = img.convert('RGB')
        mask = np.array(weighted_pseudo_mask_dict[disease])

        rgb_mask = np.zeros((rgb_img.size[0], rgb_img.size[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(rgb_mask).convert('RGB')

        mask_img.putalpha(75)
        rgb_img.paste(mask_img, (0, 0), mask_img)
        rgb_img.save(os.path.join(dest_dir, "with_weighted_pseudo_mask_" + file_name))









if __name__ == "__main__":
    chexpsert_src_img_dir = "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small"
    nih_chestxray_src_img_dir = "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled/"
    vindr_src_img_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs_rescaled_320*320"

    dest_dir = "/home/susu/Documents/MIDL_journal/journal_figures/all_speudo_bbox"


    chexpsert_imgs = {
        "Atelectasis":"/train/patient00011/study12/view1_frontal.jpg",
        "Cardiomegaly":"/train/patient00019/study4/view1_frontal.jpg",
        "Consolidation":"/train/patient00011/study2/view1_frontal.jpg",
        "Edema":"/train/patient00039/study1/view1_frontal.jpg",
        "Pleural Effusion":"/train/patient00022/study1/view1_frontal.jpg",
    }

    nih_chestxray_imgs = {
        "Atelectasis":"00000011_006.png",
        "Cardiomegaly":"00000001_001.png",
        "Consolidation":"00000044_000.png",
        "Edema":"00001549_014.png",
        "Effusion":"00000023_002.png",
    }

    vindr_imgs = {
        "Aortic enlargement":"000d68e42b71d3eac10ccc077aba07c1.png",
        "Cardiomegaly":"0007d316f756b3fa0baea2ff514ce945.png",
        "Pulmonary fibrosis":"0061cf6d35e253b6e7f03940592cc35e.png",
        "Pleural thickening":"00bcb82818ea83d6a86df241762cd7d0.png",
        "Pleural effusion":"018a2fe44c3451281ec338c27ecc9c53.png",

    }



    chexpsert_pseudo_bbox = "/home/susu/Remote_projects/tmi/data/pseudo_bboxs_chexpert.json"
    chexpsert_pseudo_mask = "/home/susu/Remote_projects/tmi/data/pseudo_masks_chexpert.json"
    chexpsert_weighted_pseudo_mask = "/home/susu/Remote_projects/tmi/data/weighted_pseudo_masks_chexpert.json"

    nih_chestxray_pseudo_bbox = "/home/susu/Remote_projects/tmi/data/pseudo_bboxs_nih.json"
    nih_chestxray_pseudo_mask = "/home/susu/Remote_projects/tmi/data/pseudo_masks_nih.json"
    nih_weighted_pseudo_mask = "/home/susu/Remote_projects/tmi/data/weighted_pseudo_masks_nih.json"

    vindr_pseudo_bbox = "/home/susu/Remote_projects/tmi/data/pseudo_bboxs_vindr.json"
    vindr_pseudo_mask = "/home/susu/Remote_projects/tmi/data/pseudo_masks_vindr.json"
    vindr_weighted_pseudo_mask = "/home/susu/Remote_projects/tmi/data/weighted_pseudo_masks_vindr.json"


    # draw_speudo_bbox(chexpsert_imgs, chexpsert_pseudo_bbox, chexpsert_src_img_dir, os.path.join(dest_dir, "chexpert"))
    # draw_speudo_mask(chexpsert_imgs, chexpsert_pseudo_mask, chexpsert_src_img_dir, os.path.join(dest_dir, "chexpert"))
    # draw_weighted_speudo_mask(chexpsert_imgs, chexpsert_weighted_pseudo_mask, chexpsert_src_img_dir, os.path.join(dest_dir, "chexpert"))

    # draw_speudo_bbox(nih_chestxray_imgs, nih_chestxray_pseudo_bbox, nih_chestxray_src_img_dir, os.path.join(dest_dir, "nih_chestxray"))
    # draw_speudo_mask(nih_chestxray_imgs, nih_chestxray_pseudo_mask, nih_chestxray_src_img_dir,
    #                  os.path.join(dest_dir, "nih_chestxray"))
    # draw_weighted_speudo_mask(chexpsert_imgs, chexpsert_weighted_pseudo_mask, chexpsert_src_img_dir, os.path.join(dest_dir, "chexpert"))
    # draw_weighted_speudo_mask(nih_chestxray_imgs, nih_weighted_pseudo_mask, nih_chestxray_src_img_dir,
    #                           os.path.join(dest_dir, "nih"))

    draw_speudo_bbox(vindr_imgs, vindr_pseudo_bbox, vindr_src_img_dir, os.path.join(dest_dir, "vindr"))
    draw_speudo_mask(vindr_imgs, vindr_pseudo_mask, vindr_src_img_dir, os.path.join(dest_dir, "vindr"))
    draw_weighted_speudo_mask(vindr_imgs, vindr_weighted_pseudo_mask, vindr_src_img_dir, os.path.join(dest_dir, "vindr"))



