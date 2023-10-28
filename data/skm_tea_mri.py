import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as tfs
from PIL import Image, ImageDraw
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.data_utils import map_image_to_intensity_range, normalize_image
from tqdm import tqdm




def to_numpy(tensor):
    """
    Converting tensor to numpy.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()



class SKMTEA(Dataset):

    def __init__(self, image_dir, df, train_diseases, transforms, img_size, with_BBox=False):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms
        self.with_BBox = with_BBox
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}

        if self.with_BBox == False:
            img_id = self.df.iloc[idx]['image_id']
            img_path = os.path.join(self.image_dir, img_id[:-4], img_id[-3:] + '.npy')
            img_data = np.load(img_path)
            # Normalize the data to the [0, 255] range
            normalized_img_data = (img_data - img_data.min()) / (img_data.max()-img_data.min()) * 255
            img = Image.fromarray(normalized_img_data.astype(np.uint8)).convert("L")
            if self.transforms is not None:
                img = self.transforms(img)  # return image in range (0,1)

            img = normalize_image(img)
            img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)

            # Get labels from the dataframe for current image
            label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
            label = np.array(label).astype(np.float32)
            data['img'] = img
            data['label'] = label

        if self.with_BBox == True:

            img_id = self.df.iloc[idx]['Image Index']
            img_path = os.path.join(self.image_dir, img_id[:-4], img_id[-3:] + '.npy')
            img_data = np.load(img_path)
            normalized_img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255
            img = Image.fromarray(normalized_img_data.astype(np.uint8)).convert("L")

            scale_factor_x = self.img_size / float(img_data.shape[0])
            scale_factor_y = self.img_size / float(img_data.shape[1])

            if self.transforms is not None:
                img = self.transforms(img)  # return image in range (0,1)

            img = normalize_image(img)
            img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)
            data['img'] = img

            label = np.zeros(len(self.TRAIN_DISEASES))
            bbox = np.zeros(4)
            lesion_type = self.df.iloc[idx]['Finding Label']
            if lesion_type in self.TRAIN_DISEASES:
                disease_idx = self.TRAIN_DISEASES.index(lesion_type)
                label[disease_idx] = 1
                x_min = int(self.df.iloc[idx]['Bbox [x'] * scale_factor_x)
                y_min = int(self.df.iloc[idx]['y'] * scale_factor_y)
                width = int(self.df.iloc[idx]['w'] * scale_factor_x)
                height = int(self.df.iloc[idx]['h]'] * scale_factor_y)
                bbox = np.array([x_min, y_min, width, height])

            data['label'] = label
            data['BBox'] = bbox

        return data



class SKMTEADataModule(LightningDataModule):

    def __init__(self, dataset_params, img_size=320, seed=42):

        self.image_dir = dataset_params["image_dir"]
        self.train_csv_file = dataset_params["train_csv_file"]
        self.valid_csv_file = dataset_params["valid_csv_file"]
        self.test_csv_file = dataset_params["test_csv_file"]
        self.BBox_csv_file = dataset_params["BBox_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]
        self.img_size = img_size
        self.seed = seed

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor()]),
            'test': tfs.Compose([tfs.Resize((self.img_size, self.img_size)),
                                 tfs.ToTensor()]),
        }

    def setup(self):

        train_df = pd.read_csv(self.train_csv_file)
        valid_df = pd.read_csv(self.valid_csv_file)
        test_df = pd.read_csv(self.test_csv_file)

        self.train_df = train_df.fillna(0)
        self.valid_df = valid_df.fillna(0)
        self.test_df = test_df.fillna(0)

        self.BBox_test_df = pd.read_csv(self.BBox_csv_file)

        self.train_set = SKMTEA(image_dir=self.image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES,
                                  transforms=self.data_transforms['train'], img_size=self.img_size)
        self.valid_set = SKMTEA(image_dir=self.image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES,
                                  transforms=self.data_transforms['test'], img_size=self.img_size)
        self.test_set = SKMTEA(image_dir=self.image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES,
                                 transforms=self.data_transforms['test'], img_size=self.img_size)

        self.BBox_test_set = SKMTEA(image_dir=self.image_dir, df=self.BBox_test_df,
                                          train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'],
                                          img_size=self.img_size, with_BBox=True)


        # To train Attri-Net, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()


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
        subset = SKMTEA(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms, img_size=self.img_size)
        return subset



    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

    def BBox_test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.BBox_test_set, batch_size=batch_size, shuffle=shuffle)

    def single_disease_train_dataloaders(self, batch_size, shuffle=True):
        train_dataloaders = {}
        for c in ['neg', 'pos']:
            train_loader = {}
            for disease in self.TRAIN_DISEASES:
                train_loader[disease] = DataLoader(self.single_disease_train_sets[c][disease], batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=True)
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







if __name__ == '__main__':

    def create_mask_fromBB(img_size, bbox):
        #bbox: [x, y, w, h]
        mask = np.zeros(img_size)
        mask[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
        return mask

    skmtea_dict = {
        "image_dir": "/mnt/qb/work/baumgartner/sun22/data/skm-tea_imgs",
        "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/train.csv",
        "valid_csv_file": "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/val.csv",
        "test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test.csv",
        "BBox_csv_file": "/mnt/qb/work/baumgartner/sun22/data/skm-tea_bbox/test_BBox.csv",
        "train_diseases": ["Meniscal Tear", "Ligament Tear", "Cartilage Lesion", "Effusion"],
    }

    data_default_params = {
        "img_size": 320,
        "batch_size": 4,
    }

    datamodule = SKMTEADataModule(skmtea_dict,
                                    img_size=data_default_params['img_size'],
                                    seed=42)


    datamodule.setup()

    train_loaders = datamodule.train_dataloader(batch_size=4)
    print('len(train_loaders.dataset)', len(train_loaders.dataset))

    valid_loaders = datamodule.valid_dataloader(batch_size=4)
    print('len(valid_loaders.dataset)',len(valid_loaders.dataset))

    test_loaders = datamodule.test_dataloader(batch_size=1)
    print('len(test_loaders.dataset)',len(test_loaders.dataset))

    BBox_test_dataloader = datamodule.BBox_test_dataloader(batch_size=1, shuffle=True)
    print('len(test_loaders.dataset)', len(BBox_test_dataloader.dataset))


    for idx in tqdm(range(len(BBox_test_dataloader.dataset))):
        print(idx)
        data = BBox_test_dataloader.dataset[idx]
        img = data['img']
        label = data['label']
        if label[3] == 1:
            bbox = data['BBox']
            BBmask = create_mask_fromBB(img_size=(320,320), bbox=bbox)

            x_min = int(bbox[0].item())
            y_min = int(bbox[1].item())
            x_max = x_min + int(bbox[2].item())
            y_max = y_min + int(bbox[3].item())

            img = Image.fromarray((np.squeeze(img)*0.5+0.5) * 255).convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
            # draw.rectangle((0, 280, 100, 290), fill=None, outline=(0, 255, 0)) # a test rectangle if not sure about x, y coordinates
            img.show()

            BBmask = Image.fromarray(BBmask*255).convert('RGB')
            BBmask.show()

    # train_dataloaders = datamodule.single_disease_train_dataloaders(batch_size=4, shuffle=False)
    # for disease in skmtea_dict["train_diseases"]:
    #     print(disease)
    #     count = 0
    #     for c in ['neg', 'pos']:
    #         print(c)
    #         disease_dataloader = train_dataloaders[c][disease]
    #         print('len(disease_dataloader.dataset)',len(disease_dataloader.dataset))
    #         count += len(disease_dataloader.dataset)
    #     print('count', count)

    # for i in range(len(test_loaders.dataset)):
    #     print(i)
    #     data = test_loaders.dataset[i]
    #     img = data['img']
    #     lbl = data['label']
    #     img = to_numpy(img).squeeze()
    #     print('img.shape', img.shape)
    #     plt.imshow(img,cmap='gray')
    #     plt.show()





#
# def get_scaled_image(
#         x: Union[torch.Tensor, np.ndarray], percentile=0.99, clip=False
# ):
#     """Scales image by intensity percentile (and optionally clips to [0, 1]).
#
#     Args:
#       x (torch.Tensor | np.ndarray): The image to process.
#       percentile (float): The percentile of magnitude to scale by.
#       clip (bool): If True, clip values between [0, 1]
#
#     Returns:
#       torch.Tensor | np.ndarray: The scaled image.
#     """
#     is_numpy = isinstance(x, np.ndarray)
#     if is_numpy:
#         x = torch.as_tensor(x)
#
#     scale_factor = torch.quantile(x, percentile)
#     x = x / scale_factor
#     if clip:
#         x = torch.clip(x, 0, 1)
#
#     if is_numpy:
#         x = x.numpy()
#
#     return x
#
#
# def plot_images(
#         images, processor=None, disable_ticks=True, titles: Sequence[str] = None,
#         ylabel: str = None, xlabels: Sequence[str] = None, cmap: str = "gray",
#         show_cbar: bool = False, overlay=None, opacity: float = 0.3,
#         hsize=5, wsize=5, axs=None
# ):
#     """Plot multiple images in a single row.
#
#     Add an overlay with the `overlay=` argument.
#     Add a colorbar with `show_cbar=True`.
#     """
#
#     def get_default_values(x, default=""):
#         if x is None:
#             return [default] * len(images)
#         return x
#
#     titles = get_default_values(titles)
#     ylabels = get_default_values(images)
#     xlabels = get_default_values(xlabels)
#
#     N = len(images)
#     if axs is None:
#         fig, axs = plt.subplots(1, N, figsize=(wsize * N, hsize))
#     else:
#         assert len(axs) >= N
#         fig = axs.flatten()[0].get_figure()
#
#     for ax, img, title, xlabel in zip(axs, images, titles, xlabels):
#         if processor is not None:
#             img = processor(img)
#         im = ax.imshow(img, cmap=cmap)
#         ax.set_title(title)
#         ax.set_xlabel(xlabel)
#
#     if overlay is not None:
#         for ax in axs.flatten():
#             im = ax.imshow(overlay, alpha=opacity)
#
#     if show_cbar:
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         fig.colorbar(im, cax=cbar_ax)
#
#     if disable_ticks:
#         for ax in axs.flatten():
#             ax.get_xaxis().set_ticks([])
#             ax.get_yaxis().set_ticks([])
#
#     return axs
#
#
# def preprocess_annotations(src_json, dest_csv):
#     # Load the JSON file
#     with open(src_json, "r") as json_file:
#         # Parse the JSON content into a Python dictionary
#         data = json.load(json_file)
#
#     categories = data["categories"]
#     images = data["images"]
#     annotations = data["annotations"]
#     print(len(images))
#     print(len(annotations))
#
#     # Create a Pandas DataFrame to store the annotations
#     # get the column names
#     column_names = ["MTR_ID", "msp_id"]
#     for i in range(len(categories)):
#         column_names.append(categories[i]["name"])
#     column_names.append("bbox")
#     print(column_names)
#     df = pd.DataFrame(0, index=np.arange(len(annotations)), columns=column_names)
#
#     # Fill the DataFrame with the annotations
#     for i in range(len(annotations)):
#         anno = annotations[i]
#         image_id = anno["image_id"]
#         category_id = anno["category_id"]
#         bbox = anno["bbox"]
#         assert category_id == categories[category_id - 1]['id']
#         disease_type = categories[category_id - 1]["name"]
#         assert image_id == images[image_id - 1]["id"]
#         img = images[image_id - 1]
#         MTR_ID = img["file_name"][:-3]
#         msp_id = img["msp_id"]
#         df.loc[i, "MTR_ID"] = MTR_ID
#         df.loc[i, "msp_id"] = msp_id
#         df.loc[i, disease_type] = 1
#         df.loc[i, "bbox"] = str(bbox)
#     df.to_csv(dest_csv, index=False)
#
#
#
# if __name__ == '__main__':
#     # root = "/mnt/qb/work/baumgartner/sun22/data/skm-tea"
#     # for dataset in ["train", "val", "test"]:
#     #     src_json = os.path.join(root, "annotations", "v1.0.0", dataset +".json")
#     #     dest_csv = os.path.join(root, "csv_annotations", dataset + ".csv")
#     #     preprocess_annotations(src_json, dest_csv)
#
#     import h5py
#     file = "/mnt/qb/work/baumgartner/sun22/data/skm-tea/test_h5file/MTR_001.h5"
#     with h5py.File(file, 'r') as data:
#         print(data.keys())
#         maps = data['maps']
#         # print(maps.shape)
#         # masks = data['masks']
#         # print(type)
#         target = data['target']
#         print(target.shape) # output: (512, 512, 160, 2, 1)
#         slice = target[:,:, 0, 0, 0]
#         print(slice.shape) # output: (512, 512)
#
#         abs_slice = np.abs(slice)
#         # scale_img = get_scaled_image(abs_slice, 0.95, clip=True)
#         # print(np.max(scale_img)) # output: 1.0
#         # print(np.min(scale_img)) # output: 0.00015
#
#         plt.imshow(abs_slice, cmap = 'gray')
#         plt.show()
#
#
#
#
#         # img_fs = data['img_fs']
#         # segm = data['segm']
#         # print(type(segm))
#
#
#
