# chexpert_dict = {
#     "image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/",
#     "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train.csv",
#     "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/valid.csv",
#     #"test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled",
#     "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert",
#     "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/test_labels.csv",
#     "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
#     "orientation": "Frontal",
#     "uncertainty": "toZero",
#     "train_augment": "previous" #"none", "random_crop", "center_crop", "color_jitter", "all",
# }

chexpert_dict = {
        "image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train.csv",
        "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/valid.csv",
        "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled",
        "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/test_labels.csv",
        "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
        "orientation": "Frontal",
        "uncertainty": "toZero",
    }




nih_chestxray_dict = {
    "image_dir": "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled", # contains all images for train, valid and test.
    "data_entry_csv_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/Data_Entry_2017.csv",
    "train_valid_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/train_val_list.txt",
    "test_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/test_list.txt",
    "BBox_csv_file_train": "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_valid_df_scaled.csv",
    "BBox_csv_file_test": "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_test_df_scaled.csv",
    # BBox_csv_file_train and BBox_csv_file_test are split from the original BBox_List_2017.csv with 40% for train and 60% for test. Bounding box annotation are already scaled to the rescaled image size of 320x320
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
}


vindr_cxr_dict = {
        "root_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection",
        "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs_rescaled_320*320",
        "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs_rescaled_320*320",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train_resized.csv",
        "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_test_resized.csv",
        "train_diseases": ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening','Pleural effusion'],
    }

contaminated_chexpert_dict = {
    "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/train.csv",
    "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/valid/valid_df.csv",
    "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/test/test_df.csv",
    "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/",
    "valid_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/valid/",
    "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/test/",
    "orientation": "Frontal",
    "uncertainty": "toZero",
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
}




data_default_params = {
    "split_ratio": 0.8,
    "resplit": False,
    "img_size": 320,
}

dataset_dict = {
    "chexpert": chexpert_dict,
    "nih_chestxray": nih_chestxray_dict,
    "vindr_cxr": vindr_cxr_dict,
    "contaminated_chexpert": contaminated_chexpert_dict,
}











