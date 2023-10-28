import numpy as np
import argparse
import os
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import shutil
import random
from tqdm import tqdm



class Contamination():
    def __init__(self, src_folder, src_csv, tgt_folder, dataset_type, contamination_type, contamination_degree, contaminated_class):
        self.src_folder = src_folder
        self.src_csv = src_csv
        self.tgt_folder = tgt_folder
        self.dataset_type = dataset_type # train, valid, test
        self.contamination_type = contamination_type
        self.contamination_degree = contamination_degree
        self.contaminated_class = contaminated_class
        self.out_df = pd.read_csv(self.src_csv)
        self.out_df["Contamination"] = np.zeros(len(self.out_df)).tolist()
        self.df = pd.read_csv(self.src_csv)
        if self.dataset_type != "test":
            self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        if os.path.exists(self.tgt_folder):
            print('folder alreay created!')
            shutil.rmtree(self.tgt_folder,ignore_errors=True)
        os.makedirs(self.tgt_folder,exist_ok=True)


    def contaminate(self):
        if self.contamination_type == "tag":
            myFont = ImageFont.truetype(os.path.join(os.path.dirname(__file__),'font', 'FreeMonoBold.ttf'), 18)
            pos_text = "CXR-ROOM1"
            pos_ps = (20, 280)
            for idx in tqdm(range(len(self.df))):
                img_path = os.path.join(self.src_folder, self.df.iloc[idx]['Path'])
                img = Image.open(img_path)
                img = img.resize((320, 320))
                label = self.df.iloc[idx][self.contaminated_class]
                if label == 1:
                    prob_pos = random.random()
                    if prob_pos < self.contamination_degree:
                        I1 = ImageDraw.Draw(img)
                        I1.text(pos_ps, pos_text, font=myFont, fill=(0))
                        self.out_df.at[idx, "Contamination"] = 1
                new_img_path = os.path.join(self.tgt_folder, self.df.iloc[idx]['Path'])
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                img.save(new_img_path)
            self.out_df.to_csv(os.path.join(self.tgt_folder, self.dataset_type+'_df.csv'),index=False)







def str2bool(v):
    return v.lower() in ('true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contaminated_class', type=str, default="Cardiomegaly",choices=["Cardiomegaly"])
    parser.add_argument('--contamination_type', type=str, default="tag", choices=["tag"])
    parser.add_argument('--contamination_scale', type=float, default=0.5, choices=[0.2, 0.5])
    return parser


def main(config):

    chexpert_dict = {
        "image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/",
        "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train.csv",
        "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/valid.csv",
        "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/chexlocalize/CheXpert/test_labels.csv",
    }
    tgt_root_folder = "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated"

    # for dataset_type in ["train", "valid", "test"]:
    #
    #     file_name = "ctm_" + dataset_type + "_df.csv"
    #     src_csv = chexpert_dict[dataset_type + "_csv_file"]
    #     src_folder = chexpert_dict["image_dir"] if dataset_type != "test" else chexpert_dict["test_image_dir"]
    #     tgt_folder = os.path.join(tgt_root_folder, dataset_type)
    #     contaminator = Contamination(src_folder, src_csv, tgt_folder, dataset_type,
    #                                  config.contamination_type, config.contamination_scale, config.contaminated_class)
    #     contaminator.contaminate()

    dataset_type = "train"
    file_name = "ctm_" + dataset_type + "_df.csv"
    src_csv = chexpert_dict[dataset_type + "_csv_file"]
    src_folder = chexpert_dict["image_dir"] if dataset_type != "test" else chexpert_dict["test_image_dir"]
    tgt_folder = os.path.join(tgt_root_folder, dataset_type)
    contaminator = Contamination(src_folder, src_csv, tgt_folder, dataset_type,
                                 config.contamination_type, config.contamination_scale, config.contaminated_class)
    contaminator.contaminate()


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    main(config)
