import os
import shutil
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import normalize_image, map_image_to_intensity_range
from torchvision.transforms.functional import InterpolationMode


class CheXpert(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = {}
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert("L")

        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)

        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=5)

        # Get labels from the dataframe for current image
        label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
        label = np.array(label)
        data['img'] = img
        data['label'] = label
        return data



class CheXpertDataModule(LightningDataModule):

    def __init__(self, dataset_params, img_size=320, seed=42):

        self.image_dir = dataset_params["image_dir"]
        self.train_csv_file = dataset_params["train_csv_file"]
        self.valid_csv_file = dataset_params["valid_csv_file"]
        self.test_image_dir = dataset_params["test_image_dir"]
        self.test_csv_file = dataset_params["test_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]
        self.orientation = dataset_params["orientation"]
        self.uncertainty = dataset_params["uncertainty"]
        self.img_size = img_size
        self.seed = seed

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor()]),
            'test': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor()]),
        }

    def setup(self):

        train_df = pd.read_csv(self.train_csv_file)
        valid_df = pd.read_csv(self.valid_csv_file)
        test_df = pd.read_csv(self.test_csv_file)

        # we only use frontal images
        self.train_df = self.preprocess_df(train_df, orientation=self.orientation, uncertainty=self.uncertainty)
        self.valid_df = self.preprocess_df(valid_df, orientation=self.orientation, uncertainty=self.uncertainty)
        self.test_BB_df = test_df # keep the original test set for bounding box prediction
        self.test_df = self.testset_orientation_filter(test_df, orientation=self.orientation)
        # self.test_df =test_df
        # for test set, we use all images to create the test set.



        self.train_set = CheXpert(image_dir=self.image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'])
        self.valid_set = CheXpert(image_dir=self.image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        self.test_set = CheXpert(image_dir=self.test_image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        # self.BBox_test_set = CheXpert(image_dir=self.test_image_dir, df=self.test_BB_df, train_diseases=self.TRAIN_DISEASES,
        #                          transforms=self.data_transforms['test'])



        # To train Attri-Net, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()





    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

    # def BBox_test_dataloader(self, batch_size, shuffle=False):
    #     return DataLoader(self.BBox_test_set, batch_size=batch_size, shuffle=False)

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

    def testset_orientation_filter(self, test_df, orientation):
        if orientation == 'Frontal':
            test_df = test_df[test_df['Path'].str.contains('frontal')]
        if orientation == 'Lateral':
            test_df = test_df[test_df['Path'].str.contains('lateral')]
        if orientation == 'all':
            test_df = test_df
        test_df = test_df.reset_index(drop=True)
        return test_df



    def preprocess_df(self, df, orientation, uncertainty):
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '')
        df = self.fillnan(df)
        df = self.uncertainty_approach(df, uncertainty)
        # select orientation
        df = self.get_orientation(df, orientation)
        df = df.reset_index(drop=True)
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

    def create_trainsets(self):
        # create positive trainset and negative trainset for each disease
        train_sets = {}
        for c in ['neg', 'pos']:
            train_set_d = {}
            for disease in self.TRAIN_DISEASES:
                train_set_d[disease] = self.subset(src_df=self.train_df, disease=disease, label=c, transforms=self.data_transforms['train'])
            train_sets[c] = train_set_d
        return train_sets


    def create_vissets(self):
        # create positive and negative visualization set for each disease
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_df=self.train_df[0:1000], disease=disease, label=c, transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets


    def subset(self, src_df, disease, label, transforms):
        """
        Create positive or negative subset from source data frame for a given disease
        :param src_df: source data frame
        :param disease: str, the specific disease to filter
        :param label: str, 'neg' for negative samples, 'pos' for positive samples
        :param transforms: torchvision.transforms
        :return: a CheXpert Dataset object
        """

        if label == 'pos':
            idx = np.where(src_df[disease] == 1)[0]
        if label == 'neg':
            idx = np.where(src_df[disease] == 0)[0]
        filtered_df = src_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = CheXpert(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset


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






if __name__ == '__main__':

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

    data_default_params = {
        "img_size": 320,
        "batch_size": 4,
    }

    datamodule = CheXpertDataModule(chexpert_dict,
                                    img_size=data_default_params['img_size'],
                                    seed=42)


    datamodule.setup()

    train_loaders = datamodule.train_dataloader(batch_size=4)
    print('len(train_loaders)', len(train_loaders))
    print('len(train_loaders.dataset)', len(train_loaders.dataset))

    valid_loaders = datamodule.valid_dataloader(batch_size=4)
    print('len(valid_loaders.dataset)',len(valid_loaders.dataset))

    test_loaders = datamodule.test_dataloader(batch_size=1)
    print('len(test_loaders.dataset)',len(test_loaders.dataset))


    # for batch in test_loaders:
    #     print(batch['img'].shape)
    #     print(batch['label'].shape)


    train_dataloaders = datamodule.single_disease_train_dataloaders(batch_size=4, shuffle=False)
    for disease in chexpert_dict["train_diseases"]:
        print(disease)
        count = 0
        for c in ['neg', 'pos']:
            print(c)
            disease_dataloader = train_dataloaders[c][disease]
            print('len(disease_dataloader.dataset)',len(disease_dataloader.dataset))
            count += len(disease_dataloader.dataset)
        print('count', count)
#
#     import matplotlib.pyplot as plt
#     def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
#         if not isinstance(imgs[0], list):
#             # Make a 2d grid even if there's just 1 row
#             imgs = [imgs]
#
#         num_rows = len(imgs)
#         num_cols = len(imgs[0]) + with_orig
#         fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#         for row_idx, row in enumerate(imgs):
#             row = row
#             for col_idx, img in enumerate(row):
#                 ax = axs[row_idx, col_idx]
#                 ax.imshow(np.asarray(img), **imshow_kwargs)
#                 ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#
#         if with_orig:
#             axs[0, 0].set(title='Original image')
#             axs[0, 0].title.set_size(8)
#         if row_title is not None:
#             for row_idx in range(num_rows):
#                 axs[row_idx, 0].set(ylabel=row_title[row_idx])
#
#         plt.tight_layout()
#         plt.show()
#
#
#     imgs = []
#     for i in range(4):
#         data = datamodule.train_set[i]
#         imgs.append(data['img'].squeeze())
#
#     plot(imgs)

