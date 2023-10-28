import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import normalize_image, map_image_to_intensity_range


class Contaminate_CheXpert(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image to return
        :return: image (PIL.Image): PIL format image
        '''
        data = {}
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert("L")

        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)

        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)

        # Get labels from the dataframe for current image
        label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
        label = np.array(label)
        data['img'] = img
        data['label'] = label

        return data



Cardiomegaly_tag_degree50_dict = {
    # "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/train.csv",
    # "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/valid/valid_df.csv",
    # "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/test/test_df.csv",
    "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/",
    "valid_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/valid/",
    "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/test/",
    "orientation": "Frontal",
    "uncertainty": "toZero",
    "train_augment": "previous",
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
}


class Contaminate_CheXpertDataModule(LightningDataModule):

    def __init__(self, dataset_params, img_size=320, seed=42):


        self.TRAIN_DISEASES = dataset_params["train_diseases"]

        self.train_dir = os.path.join(dataset_params["train_image_dir"])
        self.valid_dir = os.path.join(dataset_params["valid_image_dir"])
        self.test_dir = os.path.join(dataset_params["test_image_dir"])
        self.orientation = dataset_params["orientation"]
        self.uncertainty = dataset_params["uncertainty"]

        self.img_size = img_size
        self.seed = seed
        self.printout_statics = True

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                # tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),

            ]),
            'test': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),

            ]),
        }

    def setup(self):

        # directly read splitted df
        print('Already split data, will use previous created splitting dataframe!')
        # Note: train and validation have same contamination rate.
        train_df = pd.read_csv(os.path.join(self.train_dir, 'train_df.csv'))
        valid_df = pd.read_csv(os.path.join(self.valid_dir, 'valid_df.csv'))
        # Note that here the test set is with 100% negative samples contaminated and 0 positive samples not.
        test_df = pd.read_csv(os.path.join(self.test_dir, 'test_df.csv'))

        # we only use frontal images
        self.train_df = self.preprocess_df(train_df, orientation=self.orientation, uncertainty=self.uncertainty)
        self.valid_df = self.preprocess_df(valid_df, orientation=self.orientation, uncertainty=self.uncertainty)
        self.test_df = self.testset_orientation_filter(test_df, orientation=self.orientation)
        # for test set, we use all images to create the test set.
        # self.test_df = test_df

        if self.printout_statics:
            print("statics of train_df")
            self.print_statics(self.train_df)
            print("statics of valid_df")
            self.print_statics(self.valid_df)
            print("statics of test_df")
            self.print_statics(self.test_df)



        self.train_set = Contaminate_CheXpert(image_dir=self.train_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'])
        self.valid_set = Contaminate_CheXpert(image_dir=self.valid_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        self.test_set = Contaminate_CheXpert(image_dir=self.test_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])

        print('len(self.train_set)',len(self.train_set))
        print('len(self.valid_set)', len(self.valid_set))
        print('len(self.test_set)', len(self.test_set))

        # To train MT_VAGAN, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()

    def testset_orientation_filter(self, test_df, orientation):
        if orientation == 'Frontal':
            test_df = test_df[test_df['Path'].str.contains('frontal')]
        if orientation == 'Lateral':
            test_df = test_df[test_df['Path'].str.contains('lateral')]
        if orientation == 'all':
            test_df = test_df
        test_df = test_df.reset_index(drop=True)
        return test_df




    def get_orientation(self, df, orientation='Frontal'):
        # Deleting either lateral or frontal images of the Dataset or keep all
        if orientation == "Lateral":
            df = df[df['Frontal/Lateral'] == 'Lateral']
        elif orientation == "Frontal":
            df = df[df['Frontal/Lateral'] == 'Frontal']
        elif orientation == "all":
            df=df
        else:
            raise Exception("Wrong orientation input given!")
        return df

    def fillnan(self, df):
        new_df = df.fillna(-1)
        return new_df

    def uncertainty_approach(self, df, uncertainty):
        # uncertainty labels are mapped to 1 or 0 or kept as -1.
        if uncertainty == 'toOne':
            new_df = df.replace(-1, 1)
        if uncertainty == 'toZero':
            new_df = df.replace(-1, 0)
        if uncertainty =='keep':
            new_df = df
        return new_df

    def preprocess_df(self, df, orientation, uncertainty):
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '')
        df = self.fillnan(df)
        df = self.uncertainty_approach(df, uncertainty)
        # select orientation
        df = self.get_orientation(df, orientation)
        df = df.reset_index(drop=True)
        return df



    def print_statics(self, df):
        print("length of this dataframe: ", len(df))
        pos_samples = df[df[self.TRAIN_DISEASES[1]] == 1]
        neg_samples = df[df[self.TRAIN_DISEASES[1]] == 0]
        print("number of positive case: ", len(pos_samples))
        print("number of negative case: ", len(neg_samples))

        pos_ctm = pos_samples[pos_samples["Contamination"] == 1]
        neg_ctm = neg_samples[neg_samples["Contamination"] == 1]

        print("positive ctm ratio", len(pos_ctm) / len(pos_samples))
        print("negative ctm ratio", len(neg_ctm) / len(neg_samples))

    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)


    def single_disease_train_dataloaders(self, batch_size, shuffle=True):
        train_dataloaders = {}
        for c in ['neg', 'pos']:
            train_loader = {}
            for disease in self.TRAIN_DISEASES:
                train_loader[disease] = DataLoader(self.single_disease_train_sets[c][disease], batch_size=batch_size, shuffle=shuffle, drop_last=True)
            train_dataloaders[c] = train_loader
        return train_dataloaders

    def single_disease_vis_dataloaders(self, batch_size, shuffle=False):
        vis_dataloaders = {}
        for c in ['neg', 'pos']:
            vis_loader = {}
            for disease in self.TRAIN_DISEASES:
                vis_loader[disease] = DataLoader(self.single_disease_vis_sets[c][disease], batch_size=batch_size, shuffle=shuffle)
            vis_dataloaders[c] = vis_loader
        return vis_dataloaders


    def create_trainsets(self):
        # create positive trainset and negative trainset for each disease
        train_sets = {}
        for c in ['neg', 'pos']:
            train_set_d = {}
            for disease in self.TRAIN_DISEASES:
                train_set_d[disease] = self.subset(src_dir= self.train_dir, src_df=self.train_df, disease=disease, label=c, transforms=self.data_transforms['train'])
            train_sets[c] = train_set_d
        return train_sets


    def create_vissets(self):
        # create positive and negative visualization set for each disease
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_dir= self.train_dir, src_df=self.train_df[0:1000], disease=disease, label=c, transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets


    def subset(self, src_dir, src_df, disease, label, transforms):
        # create subset from source dataset using given selected indices
        '''
        :param src_df: source data frame
        :param disease: str, disease to filter
        :param label: str, 'neg', 'pos'
        :return: a Dataset object
        '''

        if label == 'pos':
            idx = np.where(src_df[disease] == 1)[0]
        if label == 'neg':
            idx = np.where(src_df[disease] == 0)[0]
        filtered_df = src_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = Contaminate_CheXpert(image_dir=src_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset



if __name__ == '__main__':

    # chexpert_dict = {
    #     "dataset_dir": "/mnt/qb/work/baumgartner/sun22/contaimated_data_resize/contaimated_chexpert",
    #     "train_diseases": ["Cardiomegaly"],
    # }

    # Cardiomegaly_tag_degree20_dict = {
    #     "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/train.csv",
    #     "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/valid/valid_df.csv",
    #     "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/test/test_df.csv",
    #     "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/train/",
    #     "valid_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/valid/",
    #     "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert_official_contaminated/test/",
    #     "orientation": "Frontal",
    #     "uncertainty": "toZero",
    #     "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
    # }

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
        "img_size": 320,
        "batch_size": 4,
    }

    datamodule = Contaminate_CheXpertDataModule(Cardiomegaly_tag_degree50_dict,
                                    img_size=data_default_params['img_size'],
                                    seed=42)


    datamodule.setup()
    val_loaders = datamodule.valid_dataloader(batch_size=4)
    print('len(val_loaders.dataset)',len(val_loaders.dataset))

    test_loaders = datamodule.test_dataloader(batch_size=1)
    print('len(test_loaders.dataset)',len(test_loaders.dataset))


    # for jdx, data in enumerate(test_loaders):
    #     if jdx > 4:
    #         break
    #     test_img = data['img']
    #     test_labels = data['label']
    #     print('test_img.size()',test_img.size())
    #     print('test_labels', test_labels)
    #
    # train_loaders = datamodule.single_disease_train_dataloaders(batch_size=4)
    # vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=4)
    #
    # for c in ['neg', 'pos']:
    #     for disease in chexpert_dict["train_diseases"]:
    #         print(c + ': ' + disease)
    #         print('len.dataset)', len(train_loaders[c][disease].dataset))








