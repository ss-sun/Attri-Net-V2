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

csv_file = "/mnt/qb/rawdata/vindr-cxr-physionet/1.0.0/annotations/annotations_test.csv"
df = pd.read_csv(csv_file)
print(len(df))
class_name = np.unique(df['class_name'].tolist())
print(class_name)


#
#
# class Vindr_CXR(Dataset):
#     def __init__(self, image_dir, df, train_diseases, transforms, img_size, with_BBox):
#         self.image_dir = image_dir
#         self.df = df
#         self.TRAIN_DISEASES = train_diseases
#         self.transforms = transforms
#         self.img_size = img_size
#         # create image list
#         # img_id = df['image_id'].tolist()
#         # self.image_list = np.unique(np.asarray(img_id))
#         self.with_BBox = with_BBox
#         # if self.with_BBox:
#         #     self.image_dir = ""
#
#     def __len__(self):
#         return len(self.df)
#         # return len(self.image_list)
#
#     def __getitem__(self, idx):
#
#         data = {}
#         img_id = self.df.iloc[idx]['image_id']
#         img_path = os.path.join(self.image_dir, img_id + '.png')
#         img = Image.open(img_path)  # value (0,255)
#         # note: img.size[0] is the width of the image since img is PIL image
#         # print(type(img)) # <class 'PIL.PngImagePlugin.PngImageFile'>
#         # print(img.size) # (2304, 2880)
#         # img_array = np.array(img)
#         # print(img_array.shape) # (2880, 2304)
#
#         if self.with_BBox == False:
#             # Get labels from the dataframe for current image
#             label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
#             label = np.array(label)
#             data['label'] = label
#             if self.transforms is not None:
#                 img = self.transforms(img)  # return image in range (0,1)
#             img = normalize_image(img)
#             img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)
#             data['img'] = img
#
#         if self.with_BBox == True:
#             # get the scale factor to scale the bounding box coordinates to adpat image size change from 1024 -> 320.
#
#             scale_factor_w = self.img_size / float(img.size[0])  # img.size[0] is the width of the image since img is PIL image
#             scale_factor_h = self.img_size / float(img.size[1])  # img.size[1] is the height of the image since img is PIL image. different from numpy array!
#
#             if self.transforms is not None:
#                 img = self.transforms(img)  # return image in range (0,1)
#
#             img = normalize_image(img)
#             img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)
#             data['img'] = img
#
#             label = np.zeros(len(self.TRAIN_DISEASES))
#             bbox = np.zeros(4)
#
#             lesion_type = self.df.iloc[idx]['class_name']
#             if lesion_type in self.TRAIN_DISEASES:
#                 disease_idx = self.TRAIN_DISEASES.index(lesion_type)
#                 label[disease_idx] = 1
#                 # note: the x,y coordinates of the bounding box corrspond to the top left corner of the bounding box, and the width and height of the bounding box.
#                 x_min = int(self.df.iloc[idx][
#                                 'x_min'] * scale_factor_w)  # original 'x_min' is the width coordinate of the bounding box
#                 y_min = int(self.df.iloc[idx][
#                                 'y_min'] * scale_factor_h)  # original 'y_min' is the height coordinate of the bounding box
#                 x_max = int(self.df.iloc[idx]['x_max'] * scale_factor_w)
#                 y_max = int(self.df.iloc[idx]['y_max'] * scale_factor_h)
#                 x = x_min
#                 y = y_min
#                 w = x_max - x_min
#                 h = y_max - y_min
#
#                 bbox = np.array([x, y, w, h])  # change bbox to [x_min, y_min, width, height]
#
#             data['label'] = label
#             data['BBox'] = bbox
#         return data
#
#
class Vindr_CXRDataModule(LightningDataModule):

    def __init__(self, dataset_params, split_ratio=0.8, resplit=True, img_size=320, seed=42, with_bb=False):

        self.root_dir = dataset_params["root_dir"]
        self.train_csv_file = dataset_params["train_csv_file"]
        self.test_csv_file = dataset_params["test_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]

        self.train_image_dir = self.root_dir + "/train_pngs_rescaled"
        self.BB_image_dir = dataset_params["BB_image_dir"]
        self.BBox_csv_file = dataset_params["BBox_csv_file"]
        self.split_df_dir = os.path.join(self.image_dir, 'split_df')
        self.split_ratio = split_ratio
        self.resplit = resplit
        self.img_size = img_size
        self.seed = seed
        self.with_bb = with_bb
        self.diagnoses = []

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                tfs.ToTensor(),
            ]),
            'test': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),
            ]),
        }

    def setup(self):
        # Read train and test csv files
        train_df = pd.read_csv(self.train_csv_file)
        self.diagnoses = np.unique(np.asarray(train_df['class_name'].tolist())).tolist()

        # Preprocess csv file
        # Split the train dataframe into train, valid and test dataframe
        if os.path.exists(self.split_df_dir) and self.resplit == False:
            print('Already split data, will use previous created split dataframe!')
            self.train_df = pd.read_csv(os.path.join(self.split_df_dir, 'train_cls_labels.csv'))
            self.valid_df = pd.read_csv(os.path.join(self.split_df_dir, 'valid_cls_labels.csv'))
            self.test_df = pd.read_csv(os.path.join(self.split_df_dir, 'test_cls_labels.csv'))


        else:
            if os.path.exists(self.split_df_dir):
                shutil.rmtree(self.split_df_dir)
            os.mkdir(self.split_df_dir)
            self.train_df, self.valid_df, self.test_df = self.split(df=train_df, train_ratio=self.split_ratio,
                                                                    shuffle=True)
            self.train_df, self.valid_df, self.test_df = self.restructure_cls_csv()

        self.BBox_test_df = pd.read_csv(self.BBox_csv_file)

        # Create datasets
        self.train_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES,
                                   transforms=self.data_transforms['train'], img_size=self.img_size, with_BBox=False)
        self.valid_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES,
                                   transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=False)
        self.test_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES,
                                  transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=False)
        self.BBox_test_set = Vindr_CXR(image_dir=self.BB_image_dir, df=self.BBox_test_df,
                                       train_diseases=self.TRAIN_DISEASES,
                                       transforms=self.data_transforms['test'], img_size=self.img_size,
                                       with_BBox=True)

        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()

    def BBox_test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.BBox_test_set, batch_size=batch_size, shuffle=shuffle)

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

        if self.with_bb:
            pos_idx = np.where(src_df['class_name'] == disease)[0]
        else:
            pos_idx = np.where(src_df[disease] == 1)[0]

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
        subset = Vindr_CXR(image_dir=self.train_image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES,
                           transforms=transforms, img_size=self.img_size, with_BBox=self.with_bb)

        return subset

    def split(self, df, train_ratio, shuffle=True):

        image_list = df['image_id'].tolist()
        unique_images = np.unique(np.asarray(image_list))
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(unique_images)

        split1 = int(np.floor(train_ratio * len(unique_images)))
        split2 = split1 + int(np.floor(0.5 * (1 - train_ratio) * len(unique_images)))
        train_images, valid_images, test_images = unique_images[:split1], unique_images[split1:split2], unique_images[
                                                                                                        split2:]

        train_indices = []
        valid_indices = []
        test_indices = []
        for index, row in df.iterrows():
            img_id = row['image_id']
            if img_id in train_images:
                train_indices.append(index)
            elif img_id in valid_images:
                valid_indices.append(index)
            else:
                test_indices.append(index)

        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        test_df = df.iloc[test_indices]

        # reset index to get continuous index from 0
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(self.split_df_dir, "train_df.csv"))
        valid_df.to_csv(os.path.join(self.split_df_dir, "valid_df.csv"))
        test_df.to_csv(os.path.join(self.split_df_dir, "test_df.csv"))

        return train_df, valid_df, test_df

    def restructure_cls_csv(self):
        # Rewrite the original csv files. columns are 28 labels + image_id, each row is a image.
        # Restructuring the csv files is to make it easier to create dataloader for classification task.
        train_df = self.create_df(self.train_df, 'train_cls_labels.csv')
        valid_df = self.create_df(self.valid_df, 'valid_cls_labels.csv')
        test_df = self.create_df(self.test_df, 'test_cls_labels.csv')

        return train_df, valid_df, test_df

    def create_df(self, df, filename):
        unique_img_id = np.unique(df['image_id'].tolist())
        columns = self.diagnoses
        new_df = pd.DataFrame(0.0, index=np.arange(len(unique_img_id)), columns=columns)
        new_df['image_id'] = unique_img_id

        for i, row in new_df.iterrows():
            img_id = row['image_id']
            rows = df.loc[df['image_id'] == img_id]
            for j, r in rows.iterrows():
                lesion_type = r['class_name']
                row[lesion_type] = 1.0
            new_df.loc[i] = row

        path = os.path.join(self.split_df_dir, filename)
        new_df.to_csv(path)
        return new_df
#
# # if __name__ == '__main__':
# #     from tqdm import tqdm
# #     import matplotlib.pyplot as plt
# #     from PIL import Image, ImageDraw
# #
# #     def create_mask_fromBB(img_size, bbox):
# #         # bbox: [x, y, w, h]
# #         mask = np.zeros(img_size)
# #         mask[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])] = 1
# #         return mask
# #
# #
# #     vindr_cxr_dict = {
# #         "image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection",
# #         "BB_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs",
# #         "BBox_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/splitted_df/test_df.csv",
# #         "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train.csv",
# #         "train_diseases": ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening',
# #                            'Pleural effusion'],
# #     }
# #
# #     data_default_params = {
# #         "split_ratio": 0.8,
# #         "resplit": False,
# #         "img_size": 320,
# #     }
# #
# #     datamodule = Vindr_CXRDataModule(vindr_cxr_dict,
# #                                         split_ratio=data_default_params['split_ratio'],
# #                                         resplit=data_default_params['resplit'],
# #                                         img_size=data_default_params['img_size'],
# #                                         seed=42)
# #
# #     datamodule.setup()
# #     train_loader = datamodule.train_dataloader(batch_size=4)
# #     valid_loader = datamodule.valid_dataloader(batch_size=4)
# #     test_loader = datamodule.test_dataloader(batch_size=4)
# #
# #     BBox_test_dataloader = datamodule.BBox_test_dataloader(batch_size=1, shuffle=True)
# #     print('len(test_loaders.dataset)', len(BBox_test_dataloader.dataset))
# #
# #     print(len(train_loader.dataset))
# #     print(len(valid_loader.dataset))
# #     print(len(test_loader.dataset))
# #     print(len(train_loader.dataset) + len(valid_loader.dataset) + len(test_loader.dataset))
# #
# #     for idx in tqdm(range(500)):
# #         print(idx)
# #         data = BBox_test_dataloader.dataset[idx]
# #         img = data['img']
# #         label = data['label']
# #         bbox = data['BBox']
# #         if np.sum(bbox)!=0 and label[1] == 1:
# #             print("cardiomegealy")
# #             BBmask = create_mask_fromBB(img_size=(320, 320), bbox=bbox)
# #             x_min = int(bbox[0].item())
# #             y_min = int(bbox[1].item())
# #             x_max = x_min + int(bbox[2].item())
# #             y_max = y_min + int(bbox[3].item())
# #
# #             img = Image.fromarray((np.squeeze(img) * 0.5 + 0.5) * 255).convert('RGB')
# #             draw = ImageDraw.Draw(img)
# #             draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
# #             draw.rectangle((0, 280, 100, 290), fill=None, outline=(255, 0, 0)) # a test rectangle if not sure about x, y coordinates
# #             img.show()
# #
# #             BBmask = Image.fromarray(BBmask * 255).convert('RGB')
# #             BBmask.show()
#
#
#
#
