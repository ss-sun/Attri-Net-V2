import os
import pandas as pd
from PIL import Image
import json
from pycocotools import mask
import torch
import numpy as np
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

TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

gt_seg_dir = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test.json"
src_csv = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/test_labels.csv"
test_image_dir = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert"
gt_mask_dir = "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXlocalize/gt_segmentations_test"
os.makedirs(gt_mask_dir, exist_ok=True)
test_df = pd.read_csv(src_csv)
with open(gt_seg_dir) as json_file:
    gt_dict = json.load(json_file)




def get_cxr_id(path):
    # Remove the "test/" prefix and ".jpg" suffix
    filename = path.replace("test/", "").replace(".jpg", "")
    # Split the filename into its components
    id_components = filename.split("/")
    # Join the components together with underscores
    cxr_id = "_".join(id_components)
    return cxr_id


def get_gt_mask(cxr_id, disease):
    gt_item = gt_dict[cxr_id][disease]
    gt_mask = mask.decode(gt_item)
    return gt_mask


def to_numpy(tensor):
    """
    Converting tensor to numpy.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


def draw_mask(gt_mask, src_img, outdir, prefix):
    src_img = np.asarray(src_img).squeeze()
    src_img = Image.fromarray(src_img * 255).convert('RGB')
    rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[:, :, 1] = gt_mask * 255
    rgb_mask[:, :, 2] = 0
    img = Image.fromarray(rgb_mask).convert('RGB')
    # Create a new ImageDraw object
    draw = ImageDraw.Draw(img)
    img.putalpha(50)
    src_img.paste(img, (0, 0), img)
    src_img.save(os.path.join(outdir, prefix +'.jpg'))



results = {}
IoU_results = {}

for index, row in test_df.iterrows():
    print("idx:", index)

    path = row['Path']
    cxr_id = get_cxr_id(path)
    img = Image.open(os.path.join(test_image_dir, path)).convert("L")

    for disease in TRAIN_DISEASES:
        if cxr_id in results:
            if disease in results[cxr_id]:
                print(f'Check for duplicates for {disease} for {cxr_id}')
                break
            else:
                results[cxr_id][disease] = 0
        else:
            # get ground truth binary mask
            if cxr_id not in gt_dict:
                continue
            else:
                results[cxr_id] = {}
                results[cxr_id][disease] = 0
                IoU_results[cxr_id] = {}
                IoU_results[cxr_id][disease] = 0

        gt_mask = get_gt_mask(cxr_id, disease)
        draw_mask(gt_mask, src_img=img, outdir=gt_mask_dir, prefix=f'{cxr_id}_{disease}_gt')


