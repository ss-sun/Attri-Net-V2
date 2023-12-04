chexpert_dict = {
    "image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/",
    "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train.csv",
    "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/valid.csv",
    #"test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/scaled",
    "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert",
    "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/test_labels.csv",
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
    "orientation": "Frontal",
    "uncertainty": "toZero",
    "train_augment": "previous" #"none", "random_crop", "center_crop", "color_jitter", "all",
}

# nih_chestxray_dict_old = {
#     "image_dir": "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled",
#     "withBB_image_dir": "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/images", # here use the original image, not the rescaled one. because the BBs are in the original image.
#     "data_entry_csv_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/Data_Entry_2017.csv",
#     "BBox_csv_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/BBox_List_2017.csv",
#     "train_valid_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/train_val_list.txt",
#     "test_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/test_list.txt",
#     "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
# }
nih_chestxray_dict = {
    "image_dir": "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled",
    # contains all images for train, valid and test.
    "data_entry_csv_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/Data_Entry_2017.csv",
    "train_valid_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/train_val_list.txt",
    "test_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/test_list.txt",
    # they are split from the original BBox_List_2017.csv with 40% for train and 60% for test.
    # bounding box annotation already scaled to the rescaled image size of 320x320
    "BBox_csv_file_train": "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_valid_df_scaled.csv",
    # with scale bbox to a rescaled image size of 320x320
    "BBox_csv_file_test": "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/BBox_test_df_scaled.csv",
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
}



vindr_cxr_dict = {
    "image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection",
    "BB_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs",
    "BBox_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/splitted_df/test_df.csv",
    "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train.csv",
    "train_diseases": ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion'],
}

vindr_cxr_withBBdict = {
        "root_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection",
        "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs_rescaled_320*320",
        "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs_rescaled_320*320",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train_resized.csv",
        "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_test_resized.csv",
        "train_diseases": ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening','Pleural effusion'],
    }



# contaminated_chexpert_dict = {
#     "image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small_contaminated/",
#     "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train.csv",
#     "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/valid.csv",
#     "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert",
#     "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/test_labels.csv",
#     "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
#     "orientation": "Frontal",
#     "uncertainty": "toZero",
#     "train_augment": "previous" #"none", "random_crop", "center_crop", "color_jitter", "all",
# }



Cardiomegaly_tag_degree20_dict = {
    "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated_20/train/train.csv",
    "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated_20/valid/valid_df.csv",
    "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated_20/test/test_df.csv",
    "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated_20/train/",
    "valid_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated_20/valid/",
    "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated_20/test/",
    "orientation": "Frontal",
    "uncertainty": "toZero",
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
}

Cardiomegaly_tag_degree50_dict = {
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
    "contam20": Cardiomegaly_tag_degree20_dict,
    "contam50": Cardiomegaly_tag_degree50_dict,
    "vindr_cxr_withBB":vindr_cxr_withBBdict,

}











