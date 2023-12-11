import os

cwd = os.getcwd()
pseudo_mask_dict = {
    "chexpert": os.path.join(cwd, "data/pseudo_masks_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"data/pseudo_masks_nih.json"),
    "contaminated_chexpert": os.path.join(cwd,"data/shortcut_guidance_mask.npy"),
}

pseudo_bbox_dict = {
    "chexpert": os.path.join(cwd, "data/pseudo_bboxs_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"data/pseudo_bboxs_nih.json"),
}

