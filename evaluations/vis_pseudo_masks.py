import numpy as np
import os
import json
from PIL import Image


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


tgt_dataset= "nih_chestxray"
guidance_type = "bbox"
dest_dir = "/home/susu/Documents/MIDL_journal/journal_figures/pseudo_guidance" #
dest_folder = os.path.join(dest_dir, tgt_dataset, guidance_type)
os.makedirs(dest_folder, exist_ok=True)


if guidance_type == "bbox":
    json_file_path = pseudo_bbox_dict[tgt_dataset]
elif guidance_type == "mask":
    json_file_path = pseudo_mask_dict[tgt_dataset]

with open(json_file_path, 'r') as file:
    data = json.load(file)

# Now 'data' contains the contents of the JSON file
print(data.keys())

if guidance_type == "bbox":
    for key in data.keys():
        bbox = data[key]
        x_min = bbox[0]
        y_min = bbox[1]
        width = bbox[2]
        height = bbox[3]
        x_max = x_min + width
        y_max = y_min + height
        mask = np.zeros((320, 320))
        mask[y_min:y_max, x_min:x_max] = 1
        BBmask = Image.fromarray((mask * 255).astype(np.uint8))
        BBmask.show()
        file_name = key +"_"+ guidance_type +".png"
        BBmask.save(os.path.join(dest_folder, file_name))


if guidance_type == "mask":
    for key in data.keys():
        mask = np.array(data[key])
        BBmask = Image.fromarray((mask * 255).astype(np.uint8))
        BBmask.show()
        file_name = key +"_"+ guidance_type +".png"
        BBmask.save(os.path.join(dest_folder, file_name))