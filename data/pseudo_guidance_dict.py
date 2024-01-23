import os

cwd = os.getcwd()
pseudo_mask_dict = {
    "chexpert": os.path.join(cwd, "data/pseudo_masks_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"data/new_pseudo_masks_nih.json"),
    "vindr_cxr": os.path.join(cwd,"data/pseudo_masks_vindr.json"),
    "vindr_cxr_mix": os.path.join(cwd,"data/pseudo_masks_vindr.json"),
    # "contaminated_chexpert": os.path.join(cwd,"data/shortcut_guidance_mask.npy"), # before
    "chexpert_mix": os.path.join(cwd, "data/pseudo_masks_chexpert.json"),
    "contaminated_chexpert": os.path.join(cwd, "data/new_shortcut_guidance_mask.npy"),
}

pseudo_bbox_dict = {
    "chexpert": os.path.join(cwd, "data/pseudo_bboxs_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"data/new_pseudo_bboxs_nih.json"),
    "vindr_cxr": os.path.join(cwd,"data/pseudo_bboxs_vindr.json"),
    "vindr_cxr_mix": os.path.join(cwd,"data/pseudo_bboxs_vindr.json"),
    "chexpert_mix": os.path.join(cwd, "data/pseudo_bboxs_chexpert.json"),
}

weighted_pseudo_mask_dict = {
    "chexpert": os.path.join(cwd, "data/weighted_pseudo_masks_chexpert.json"),
    "nih_chestxray": os.path.join(cwd,"data/new_weighted_pseudo_masks_nih.json"),
    "vindr_cxr": os.path.join(cwd,"data/weighted_pseudo_masks_vindr.json"),
    "vindr_cxr_mix": os.path.join(cwd,"data/weighted_pseudo_masks_vindr.json"),
    "chexpert_mix": os.path.join(cwd, "data/weighted_pseudo_masks_chexpert.json"),
}