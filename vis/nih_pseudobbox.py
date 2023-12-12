import pandas as pd
import os
import numpy as np

from PIL import Image
from PIL import ImageDraw


def get_bbox_list(src_annotation, disease_name):
    bbox_list = []
    df = pd.read_csv(src_annotation)
    for index, row in df.iterrows():
        if row["Finding Label"] == disease_name:
            x = row["Bbox [x"]
            y = row["y"]
            w = row["w"]
            h = row["h]"]
            bbox = [x,y, w, h]
            bbox_list.append(bbox)
    return bbox_list



def draw_bbox(src_img_path, bbox_list, out_dir):

    img = Image.open(src_img_path)
    rgb_img = img.convert('RGB')

    rgb_mask = np.zeros((rgb_img.size[0], rgb_img.size[1], 3), dtype=np.uint8)
    mask_img = Image.fromarray(rgb_mask).convert('RGB')

    draw = ImageDraw.Draw(mask_img)
    outline_color = (0, 255, 0)
    outline_width = 2

    for bbox in bbox_list:
        x_min = bbox[0]
        y_min = bbox[1]
        width = bbox[2]
        height = bbox[3]
        draw.rectangle([x_min, y_min, x_min + width, y_min + height], outline=outline_color, width=outline_width)
    del draw

    mask_img.putalpha(50)
    rgb_img.paste(mask_img, (0, 0), mask_img)

    file_name = os.path.basename(src_img_path)
    path = os.path.join(out_dir, "bbox_" + file_name)
    rgb_img.save(path)





def draw_speudo_masks(src_img_path, bbox_list, out_dir):

    img = Image.open(src_img_path)
    rgb_img = img.convert('RGB')

    mask = np.zeros((rgb_img.size[0], rgb_img.size[1]))

    for bbox in bbox_list:
        x_min = bbox[0]
        y_min = bbox[1]
        width = bbox[2]
        height = bbox[3]
        mask[y_min:y_min+height, x_min:x_min+width] = 1
    rgb_mask = np.zeros((rgb_img.size[0], rgb_img.size[1], 3), dtype=np.uint8)
    rgb_mask[:,:,1] = mask * 255
    mask_img = Image.fromarray(rgb_mask).convert('RGB')

    mask_img.putalpha(50)
    rgb_img.paste(mask_img, (0, 0), mask_img)

    file_name = os.path.basename(src_img_path)
    path = os.path.join(out_dir, "pseudo_mask_" + file_name)
    rgb_img.save(path)

def draw_speudo_bbox(src_img_path, bbox_list, out_dir):
    img = Image.open(src_img_path)
    rgb_img = img.convert('RGB')

    mask = np.zeros((rgb_img.size[0], rgb_img.size[1]))
    for bbox in bbox_list:
        x_min = bbox[0]
        y_min = bbox[1]
        width = bbox[2]
        height = bbox[3]
        mask [y_min:y_min + height, x_min:x_min + width] = 1
    x_min = np.where(mask > 0)[1].min()
    y_min = np.where(mask > 0)[0].min()
    x_max = np.where(mask > 0)[1].max()
    y_max = np.where(mask > 0)[0].max()

    rgb_mask = np.zeros((rgb_img.size[0], rgb_img.size[1], 3), dtype=np.uint8)
    mask_img = Image.fromarray(rgb_mask).convert('RGB')

    draw = ImageDraw.Draw(mask_img)
    outline_color = (0, 255, 0)
    outline_width = 5

    draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color, width=outline_width)
    del draw


    mask_img.putalpha(50)
    rgb_img.paste(mask_img, (0, 0), mask_img)

    file_name = os.path.basename(src_img_path)
    path = os.path.join(out_dir, "pseudo_bbox_" + file_name)
    rgb_img.save(path)


if __name__ == "__main__":

    src_img = "00007551_020.png" # nih bbox example with cardiomegaly positive
    src_dir = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/images_scaled"
    out_dir = "/home/susu/Documents/MIDL_journal/journal_figures/speudo_bbox"
    scaled_bbox_annotation = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_valid_df_scaled.csv"
    bbox_list = get_bbox_list(scaled_bbox_annotation, disease_name="Cardiomegaly")
    draw_bbox(os.path.join(src_dir,src_img), bbox_list, out_dir)
    draw_speudo_masks(os.path.join(src_dir, src_img), bbox_list, out_dir)
    draw_speudo_bbox(os.path.join(src_dir, src_img), bbox_list, out_dir)
