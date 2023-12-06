import os
import shutil
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import map_image_to_intensity_range, normalize_image
from PIL import Image, ImageDraw




class Vindr_CXR_BBOX_TEST(Dataset):
    # this dataset is only used for pixel sensitivity analysis.
    # the diffierence between this dataset and Vindr_CXR_BBOX is that this dataset does not provide bbox for multi-labels.
    # it is consistent with the original Vindr_CXR dataset and NIH bbox data with format img_id, class_name, x_min, y_min, x_max, y_max.

    def __init__(self, image_dir, df, train_diseases, transforms, img_size, lbl_style="single-label"):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms
        self.img_size = img_size
        self.lbl_type = lbl_style

    def __len__(self):
        return len(self.df) # length of the dataframe, but not the total number of unique images. because one image may have multiple labels.

    def __getitem__(self, idx):

        data = {}
        if self.lbl_type == "single-label":
            img_id = self.df.iloc[idx]['image_id']
            img_path = os.path.join(self.image_dir, img_id + '.png')
            img = Image.open(img_path)  # value (0,255)
            if self.transforms is not None:
                img = self.transforms(img)  # return image in range (0,1)
            img = normalize_image(img)
            img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)
            data['img'] = img

            label = np.zeros(len(self.TRAIN_DISEASES))
            bbox = np.zeros(4)

            lesion_type = self.df.iloc[idx]['class_name']
            if lesion_type in self.TRAIN_DISEASES:
                disease_idx = self.TRAIN_DISEASES.index(lesion_type)
                label[disease_idx] = 1
                # note: the x,y coordinates of the bounding box corrspond to the top left corner of the bounding box, and the width and height of the bounding box.
                x_min = int(self.df.iloc[idx]['x_min'])  # original 'x_min' is the width coordinate of the bounding box
                y_min = int(self.df.iloc[idx]['y_min'])  # original 'y_min' is the height coordinate of the bounding box
                x_max = int(self.df.iloc[idx]['x_max'])
                y_max = int(self.df.iloc[idx]['y_max'])
                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min
                bbox = np.array([x, y, w, h])  # change bbox to [x_min, y_min, width, height]

            data['label'] = label
            data['BBox'] = bbox
        return data



class Vindr_CXR_BBOX(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms, img_size):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms
        self.img_size = img_size
        # create image list
        img_id = df['image_id'].tolist()
        self.image_list = np.unique(np.asarray(img_id))

    def __len__(self):
        # total number of unique images. suitable for multi-label classification and guidance.
        return len(self.image_list)

    def __getitem__(self, idx):

        data = {}

        img_id = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_id + '.png')
        img = Image.open(img_path)  # value (0,255)
        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)
        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)
        data['img'] = img

        # Get labels from the dataframe for current image
        rows = self.df.loc[self.df['image_id'] == img_id]
        # lesion_types = rows['class_name'].tolist()
        label = np.zeros(len(self.TRAIN_DISEASES))
        bbox = np.zeros((len(self.TRAIN_DISEASES), 4))
        for index, row in rows.iterrows():
            lesion_type = row['class_name']
            if lesion_type in self.TRAIN_DISEASES:
                idx = self.TRAIN_DISEASES.index(lesion_type)
                label[idx] = 1
                bb = [int(row['x_min']), int(row['y_min']), int((row['x_max']-row['x_min'])), int((row['y_max']-row['y_min']))]
                bbox[idx] = np.array(bb)

        data['img'] = img
        data['label'] = label
        data['BBox'] = bbox
        return data






class Vindr_CXR_BB_DataModule(LightningDataModule):

    def __init__(self, dataset_params, split_ratio=0.8, resplit=True, img_size=320, seed=42):

        self.root_dir = dataset_params["root_dir"]
        self.train_image_dir = dataset_params["train_image_dir"]
        self.test_image_dir = dataset_params["test_image_dir"]
        self.train_csv_file = dataset_params["train_csv_file"]
        self.test_csv_file = dataset_params["test_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]
        self.split_df_dir = os.path.join(self.root_dir, 'BBox_split_df')
        self.split_ratio = split_ratio
        self.resplit = resplit
        self.img_size = img_size
        self.seed = seed
        self.diagnoses = []

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.ToTensor(),
            ]),
            'test': tfs.Compose([
                tfs.ToTensor(),
            ]),
        }

    def setup(self):
        # Read train and test csv files
        train_df = pd.read_csv(self.train_csv_file)
        test_df = pd.read_csv(self.test_csv_file)
        print('len(train_df): ',len(train_df))
        print('len(test_df): ',len(test_df))
        self.diagnoses = np.unique(np.asarray(train_df['class_name'].tolist())).tolist()

        # Preprocess csv file
        # Split the train dataframe into train, valid dataframe
        if os.path.exists(self.split_df_dir) and self.resplit == False:
            print('Already split data, will use previous created split dataframe!')
            self.train_df = pd.read_csv(os.path.join(self.split_df_dir, 'train.csv'))
            self.valid_df = pd.read_csv(os.path.join(self.split_df_dir, 'valid.csv'))
        else:
            if os.path.exists(self.split_df_dir):
                shutil.rmtree(self.split_df_dir)
            os.mkdir(self.split_df_dir)
            self.train_df, self.valid_df = self.split(df=train_df, train_ratio=self.split_ratio, shuffle=True)

        self.test_df = test_df
        print('len(self.train_df): ', len(self.train_df))
        print('len(self.valid_df): ', len(self.valid_df))
        print('len(self.test_df): ', len(self.test_df))




        # Create datasets
        self.train_set = Vindr_CXR_BBOX(image_dir=self.train_image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES,
                                   transforms=self.data_transforms['train'], img_size=self.img_size)
        self.valid_set = Vindr_CXR_BBOX(image_dir=self.train_image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES,
                                   transforms=self.data_transforms['test'], img_size=self.img_size)
        self.test_set = Vindr_CXR_BBOX(image_dir=self.test_image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES,
                                  transforms=self.data_transforms['test'], img_size=self.img_size)

        self.bbox_test_set = Vindr_CXR_BBOX_TEST(image_dir=self.test_image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES,
                                  transforms=self.data_transforms['test'], img_size=self.img_size, lbl_style = "single-label")

        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()


    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)

    def BBox_test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.bbox_test_set, batch_size=batch_size, shuffle=False)

    def single_disease_train_dataloaders(self, batch_size, shuffle=True):
        train_dataloaders = {}
        for c in ['neg', 'pos']:
            train_loader = {}
            for disease in self.TRAIN_DISEASES:
                train_loader[disease] = DataLoader(self.single_disease_train_sets[c][disease], batch_size=batch_size,
                                                   shuffle=shuffle)
            train_dataloaders[c] = train_loader
        return train_dataloaders

    def single_disease_vis_dataloaders(self, batch_size, shuffle=False):
        vis_dataloaders = {}
        for c in ['neg', 'pos']:
            vis_loader = {}
            for disease in self.TRAIN_DISEASES:
                vis_loader[disease] = DataLoader(self.single_disease_vis_sets[c][disease], batch_size=batch_size,
                                                 shuffle=shuffle)
            vis_dataloaders[c] = vis_loader
        return vis_dataloaders

    def create_trainsets(self):
        """
        Create positive trainset and negative trainset for each disease
        """
        train_sets = {}
        for c in ['neg', 'pos']:
            train_set_d = {}
            for disease in self.TRAIN_DISEASES:
                train_set_d[disease] = self.subset(src_df=self.train_df, disease=disease, label=c,
                                                   transforms=self.data_transforms['train'])
            train_sets[c] = train_set_d
        return train_sets

    def create_vissets(self):
        """
        Create positive trainset and negative visualization for each disease
        """
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_df=self.train_df[0:2000], disease=disease, label=c,
                                                 transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets

    def subset(self, src_df, disease, label, transforms):
        pos_idx = np.where(src_df['class_name'] == disease)[0]
        pos_img_list = src_df.iloc[pos_idx]['image_id'].tolist()
        unique_pos_img = np.unique(np.asarray(pos_img_list))

        if label == 'pos':
            filter_indices = []
            for index, row in src_df.iterrows():
                image_id = row['image_id']
                if image_id in unique_pos_img:
                    filter_indices.append(index)

        if label == 'neg':
            filter_indices = []
            for index, row in src_df.iterrows():
                image_id = row['image_id']
                if image_id in unique_pos_img:
                    pass
                else:
                    filter_indices.append(index)

        filtered_df = src_df.iloc[filter_indices]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = Vindr_CXR_BBOX(image_dir=self.train_image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES,
                           transforms=transforms, img_size=self.img_size)
        return subset

    def split(self, df, train_ratio, shuffle=True):
        image_list = df['image_id'].tolist()
        unique_images = np.unique(np.asarray(image_list))
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(unique_images)

        split1 = int(np.floor(train_ratio * len(unique_images)))
        train_images, valid_images = unique_images[:split1], unique_images[split1:]
        train_indices = []
        valid_indices = []

        for index, row in df.iterrows():
            img_id = row['image_id']
            if img_id in train_images:
                train_indices.append(index)
            else:
                valid_indices.append(index)

        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        # reset index to get continuous index from 0
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        train_df.to_csv(os.path.join(self.split_df_dir, "train.csv"))
        valid_df.to_csv(os.path.join(self.split_df_dir, "valid.csv"))

        return train_df, valid_df





if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    def create_mask_fromBB(img_size, bbox):
        # bbox: [x, y, w, h]
        mask = np.zeros(img_size)
        mask[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])] = 1
        return mask


    vindr_cxr_withBBdict = {
        "root_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection",
        "train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs_rescaled_320*320",
        "test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs_rescaled_320*320",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train_resized.csv",
        "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_test_resized.csv",
        "train_diseases": ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening',
                           'Pleural effusion'],
    }

    data_default_params = {
        "split_ratio": 0.8,
        "resplit": False,
        "img_size": 320,
    }

    datamodule = Vindr_CXR_BB_DataModule(vindr_cxr_withBBdict,
                                        split_ratio=data_default_params['split_ratio'],
                                        resplit=data_default_params['resplit'],
                                        img_size=data_default_params['img_size'],
                                        seed=42)

    datamodule.setup()
    train_loader = datamodule.train_dataloader(batch_size=4)
    valid_loader = datamodule.valid_dataloader(batch_size=4)
    test_loader = datamodule.test_dataloader(batch_size=4)

    print("len(train_loader.dataset) in main", len(train_loader.dataset))
    print("len(valid_loader.dataset) in main", len(valid_loader.dataset))
    print("len(test_loader.dataset) in main", len(test_loader.dataset))

    # train_dataloaders = datamodule.single_disease_train_dataloaders(batch_size=4, shuffle=False)
    #
    # for disease in vindr_cxr_withBBdict["train_diseases"]:
    #     print(disease)
    #     count = 0
    #     for c in ['neg', 'pos']:
    #         print(c)
    #         disease_dataloader = train_dataloaders[c][disease]
    #         print('len(disease_dataloader.dataset)_' + c  , len(disease_dataloader.dataset))
    #         count += len(disease_dataloader.dataset)
    #     print('count', count)
    #
    # loader = train_dataloaders['pos']['Cardiomegaly']
    # # loader = test_loader
    #
    # for idx in tqdm(range(500)):
    #     print(idx)
    #     data = loader.dataset[idx]
    #     img = data['img']
    #     label = data['label']
    #     bbox = data['BBox'][1].astype(int)
    #     print("label: ", label)
    #     if np.sum(bbox)!=0 and label[1] == 1:
    #         print("cardiomegealy")
    #         BBmask = create_mask_fromBB(img_size=(320, 320), bbox=bbox)
    #         x_min = int(bbox[0].item())
    #         y_min = int(bbox[1].item())
    #         x_max = x_min + int(bbox[2].item())
    #         y_max = y_min + int(bbox[3].item())
    #
    #         img = Image.fromarray((np.squeeze(img) * 0.5 + 0.5) * 255).convert('RGB')
    #         draw = ImageDraw.Draw(img)
    #         draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
    #         draw.rectangle((0, 280, 100, 290), fill=None, outline=(255, 0, 0)) # a test rectangle if not sure about x, y coordinates
    #         img.show()
    #
    #         BBmask = Image.fromarray(BBmask * 255).convert('RGB')
    #         BBmask.show()
# #
# #
# #
# #

# csv_file = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train.csv"
# img_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs"
# df = pd.read_csv(csv_file)
# print(len(df))
# img_id = df['image_id'].unique()
# print(len(img_id))
# img_id = "9a5094b2563a1ef3ff50dc5c7ff71345"
# img_path = os.path.join(img_dir, img_id + '.png')
# img = Image.open(img_path).convert('RGB')  # value (0,255)
# a = df.loc[df['image_id'] == img_id]['x_min']
# x_min = int(df.loc[df['image_id'] == img_id].iloc[0]['x_min'])
# y_min = int(df.loc[df['image_id'] == img_id].iloc[0]['y_min'])
# x_max = int(df.loc[df['image_id'] == img_id].iloc[0]['x_max'])
# y_max = int(df.loc[df['image_id'] == img_id].iloc[0]['y_max'])
#
#
# draw = ImageDraw.Draw(img)
# draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
# img.show()
