import pandas as pd
import os
import numpy as np

from PIL import Image
from PIL import ImageDraw



def draw_guidance_tag(src_img_path, contaminated_src_image_path, bbox, out_dir):
    file_name = src_img_path.split("/")[-3] + "_" + src_img_path.split("/")[-2] + "_" + src_img_path.split("/")[-1]
    src = Image.open(src_img).resize((320,320)).convert('RGB')
    contam_src = Image.open(contaminated_src_image_path).convert('RGB')
    contam_src1 = Image.open(contaminated_src_image_path).convert('RGB')
    contam_src2 = Image.open(contaminated_src_image_path).convert('RGB')

    bbox_mask = np.zeros((contam_src1.size[0], contam_src1.size[1], 3), dtype=np.uint8)
    bbox_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0]+bbox[2], 0] = 255
    mask_img = Image.fromarray(bbox_mask).convert('RGB')

    mask_img.putalpha(50)
    contam_src1.paste(mask_img, (0, 0), mask_img)
    contam_src1.show()

    guidance_mask = np.zeros((contam_src2.size[0], contam_src2.size[1], 3), dtype=np.uint8)
    guidance_mask[20:295,20:295,1] = 1 * 255

    guidance_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0]+bbox[2], 1] = 0
    guidance_image = Image.fromarray(guidance_mask).convert('RGB')

    guidance_image.putalpha(50)
    contam_src2.paste(guidance_image, (0, 0), mask_img)
    contam_src2.show()

    src.save(os.path.join(out_dir, "src_" + file_name))
    contam_src.save(os.path.join(out_dir, "contam_src_" + file_name))
    contam_src1.save(os.path.join(out_dir, "contam_src_bbox_" + file_name))
    contam_src2.save(os.path.join(out_dir, "contam_src_guidance_" + file_name))





if __name__ == "__main__":
    src_img = "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train/patient00009/study1/view1_frontal.jpg"
    contaminated_src_image = "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/train/patient00009/study1/view1_frontal.jpg"
    bbox = [20, 280, 100, 18] # [x_min, y_min, width, height] of the tag we added before
    out_dir = "/home/susu/Documents/MIDL_journal/journal_figures/contaminated_guidance"
    draw_guidance_tag(src_img, contaminated_src_image, bbox, out_dir)
