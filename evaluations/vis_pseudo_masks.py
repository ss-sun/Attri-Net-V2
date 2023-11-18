import numpy as np


pseudo_mask_dict = {
    "chexpert": "/home/susu/Remote_projects/tmi/data/pseudo_masks_chexpert.json",
    "nih_chestxray": "/home/susu/Remote_projects/tmi/data/pseudo_masks_nih.json",
}

pseudo_bbox_dict = {
    "chexpert": "/home/susu/Remote_projects/tmi/data/pseudo_bboxs_chexpert.json",
    "nih_chestxray": "/home/susu/Remote_projects/tmi/data/pseudo_bboxs_nih.json",
}


chexpert_disease_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
nih_chestxray_disease_list = ['Atelectasis', 'Cardiomegaly', 'Effusion']

