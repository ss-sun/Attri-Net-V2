import os
import shutil
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import normalize_image, map_image_to_intensity_range
import matplotlib.pyplot as plt

class AIROGS_color_fundus(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = {}
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['challenge_id']+'.jpg')
        img = Image.open(img_path)
        # plt.imshow(img)
        # plt.show()
        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)
        # img = np.mean(np.array(img),axis=0, keepdims=True)
        # img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)
        # Get labels from the dataframe for current image
        label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
        label = np.asfarray(label)
        data['img'] = img
        data['label'] = label
        return data





class AIROGS_color_fundusDataModule(LightningDataModule):

    def __init__(self, dataset_params, split_ratio=0.8, resplit=False, img_size=320, seed=42):
        self.root_dir = dataset_params["root_dir"]
        # self.image_dir = os.path.join(self.root_dir, 'train')
        self.image_dir = os.path.join(self.root_dir, 'train_scaled_336')
        self.train_csv_file = dataset_params["train_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]
        self.split_ratio = split_ratio
        self.resplit = resplit
        self.img_size = img_size
        self.seed = seed
        self.split_df_dir = os.path.join(self.root_dir, 'split_df')

        # Define the mean and standard deviation values
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        self.data_transforms = {
            'train': tfs.Compose([ # I want to add scale here, because the fundus images vary in size, some are bigger
                # tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                # tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                # tfs.Resize(336), images already resized to 336 in preprocessing, so no need to resize again
                tfs.CenterCrop(320),
                # tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),
                # tfs.Normalize(mean=mean, std=std)  # Normalize the tensor

            ]),
            'test': tfs.Compose([
                # tfs.Resize(336), images already resized to 336 in preprocessing, so no need to resize again
                tfs.CenterCrop(320),
                # tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),
                # tfs.Normalize(mean=mean, std=std)  # Normalize the tensor

            ]),
        }

    def setup(self):

        # store split dataframe folder
        if os.path.exists(self.split_df_dir) and self.resplit==False:
            # directly read splitted data frame
            print('Already split data. Will use the previously created split data frame!')
            self.train_df = pd.read_csv(os.path.join(self.split_df_dir, 'train_df.csv'))
            self.valid_df = pd.read_csv(os.path.join(self.split_df_dir, 'valid_df.csv'))
            self.test_df = pd.read_csv(os.path.join(self.split_df_dir, 'test_df.csv'))

        else:
            if os.path.exists(self.split_df_dir):
                shutil.rmtree(self.split_df_dir)
            os.mkdir(self.split_df_dir)
            # first read train and test csv files
            df = pd.read_csv(self.train_csv_file)
            train_df = self.preprocess_df(df)
            # split the train dataframe into train and valid dataframe, and save to csv file
            self.train_df, self.valid_df, self.test_df = self.split(df=train_df, train_ratio=self.split_ratio, seed=self.seed, shuffle=True)

        self.train_set = AIROGS_color_fundus(image_dir=self.image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'])
        self.valid_set = AIROGS_color_fundus(image_dir=self.image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        self.test_set = AIROGS_color_fundus(image_dir=self.image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])

        # To train Attri-Net, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()

    def preprocess_df(self, df):
        df['RG'] = np.nan
        df['RG'] = df['class'].apply(lambda x: 1 if x == 'RG' else 0)
        df = df.fillna(0)
        return df

    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, concat_testset=False, batch_size=1, shuffle=False):
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
        subset = AIROGS_color_fundus(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset


    def split(self, df, train_ratio, seed, shuffle=True):
        print("spliting data frame into trainset, validation set and test set....")
        path = np.array(df["challenge_id"].tolist())
        # patient_id = [p.split("/")[1] for p in path]
        # unique_patient = np.unique(np.array(patient_id))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(path)

        split1 = int(np.floor(train_ratio * len(path)))
        split2 = split1 + int(np.floor(0.5 * (1 - train_ratio) * len(path)))
        train_patientID, valid_patientID, test_patientID = path[:split1], path[split1:split2], path[split2:]

        train_indices = []
        valid_indices = []
        test_indices = []
        for index, row in df.iterrows():
            challenge_id = row["challenge_id"]
            if challenge_id in train_patientID:
                train_indices.append(index)
            elif challenge_id in valid_patientID:
                valid_indices.append(index)
            else:
                test_indices.append(index)

        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        test_df = df.iloc[test_indices]

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(self.split_df_dir, 'train_df.csv'), index=False)
        valid_df.to_csv(os.path.join(self.split_df_dir, 'valid_df.csv'), index=False)
        test_df.to_csv(os.path.join(self.split_df_dir, 'test_df.csv'), index=False)

        return train_df, valid_df, test_df





if __name__ == '__main__':

    airogs_fundus_dict = {
        "root_dir": "/mnt/qb/work/baumgartner/sun22/data/5793241",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/5793241/train_labels.csv",
        "train_diseases": ["RG"],
    }

    data_default_params = {
        "img_size": 320,
        "batch_size": 4,
    }

    datamodule = AIROGS_color_fundusDataModule(airogs_fundus_dict,
                                    img_size=data_default_params['img_size'],
                                    seed=42)


    datamodule.setup()

    train_loaders = datamodule.train_dataloader(batch_size=4)
    print('len(train_loaders.dataset)', len(train_loaders.dataset))

    valid_loaders = datamodule.valid_dataloader(batch_size=4)
    print('len(valid_loaders.dataset)',len(valid_loaders.dataset))

    test_loaders = datamodule.test_dataloader(batch_size=1)
    print('len(test_loaders.dataset)',len(test_loaders.dataset))



    train_dataloaders = datamodule.single_disease_train_dataloaders(batch_size=4, shuffle=False)
    for disease in airogs_fundus_dict["train_diseases"]:
        print(disease)
        count = 0
        for c in ['neg', 'pos']:
            print(c)
            disease_dataloader = train_dataloaders[c][disease]
            print('len(disease_dataloader.dataset)',len(disease_dataloader.dataset))
            count += len(disease_dataloader.dataset)
        print('count', count)

    for i in range(10):
        print(i)
        data = test_loaders.dataset[i]
        img = data['img']
        lbl = data['label']
        img = img.squeeze().permute(1, 2, 0).numpy()
        print('img.shape', img.shape)
        plt.imshow(img)
        plt.show()