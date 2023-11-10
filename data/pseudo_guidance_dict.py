import os

cwd = os.getcwd()
pseudo_mask_dict = {
    "chexpert": os.path.join(cwd, "./data/pseudo_masks_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"./data/pseudo_masks_nih.json"),
}

pseudo_bbox_dict = {
    "chexpert": os.path.join(cwd, "./data/pseudo_bboxs_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"./data/pseudo_bboxs_nih.json"),
}

