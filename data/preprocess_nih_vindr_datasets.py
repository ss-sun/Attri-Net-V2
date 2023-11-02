from PIL import Image
import os
import shutil
import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
from PIL import Image, ImageDraw


# This file scales images from NIH chest X-ray and Vindr-CXR datasets.
# Images from these two datasets have large dimensions that affect the training speed if we scale during training.

def preprocess(src_dir, dest_dir, basesize=320): # rescale the image so that the minimum dimension is 320, will keep the original aspect ratio
    # Clear dest_dir
    try:
        shutil.rmtree(dest_dir)
    except:
        pass
    os.makedirs(dest_dir)
    file_lists = os.listdir(src_dir)
    for file in file_lists:
        if file.endswith(('.jpg', '.png', 'jpeg')):
            src_img_path = src_dir + file
            img = Image.open(src_img_path)
            (width, height) = img.size[-2:]
            scale = basesize/float(min(width, height))
            img = img.resize((max(basesize, int(width * scale)), max(basesize, int(height * scale))), Image.LANCZOS)
            img.save(dest_dir + file)

def rescale_imgs(src_dir, dest_dir, basesize=(320, 320)): # rescale the image to 320*320, may change the image aspect ratio
    # Clear dest_dir
    try:
        shutil.rmtree(dest_dir)
    except:
        pass
    os.makedirs(dest_dir)
    file_lists = os.listdir(src_dir)
    for file in file_lists:
        if file.endswith(('.jpg', '.png', 'jpeg')):
            src_img_path = src_dir + file
            img = Image.open(src_img_path)
            img = img.resize((basesize), Image.LANCZOS)
            img.save(dest_dir + file)




def resize_bbox_annotation(img_dir, src_csv, img_size=320):
    src_df = pd.read_csv(src_csv)
    src_df.fillna(0, inplace=True)
    new_df = pd.DataFrame(columns=src_df.columns)
    for i in tqdm(range(len(src_df))):
        row = src_df.iloc[i]
        img_path = os.path.join(img_dir, row['image_id'] + ".png")
        img = Image.open(img_path)
        (width, height) = img.size[-2:]
        scale_factor_w = img_size / float(width)  # img.size[0] is the width of the image since img is PIL image
        scale_factor_h = img_size / float(height)  # img.size[1] is the height of the image since img is PIL image. different from numpy array!
        # note: the x,y coordinates of the bounding box corrspond to the top left corner of the bounding box, and the width and height of the bounding box.
        x_min = int(row['x_min'] * scale_factor_w)  # original 'x_min' is the width coordinate of the bounding box
        y_min = int(row['y_min'] * scale_factor_h)  # original 'y_min' is the height coordinate of the bounding box
        x_max = int(row['x_max'] * scale_factor_w)
        y_max = int(row['y_max'] * scale_factor_h)

        # if row['class_name'] == "Cardiomegaly":
        #     img = img.resize((320,320), Image.LANCZOS)
        #     img = img.convert('RGB')
        #     draw = ImageDraw.Draw(img)
        #     draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
        #     img.show()
        if "test" in src_csv:
            new_df.loc[i] = [row['image_id'], row['class_name'], x_min, y_min, x_max, y_max]
        if "train" in src_csv:
            new_df.loc[i] = [row['image_id'], row['rad_id'] ,row['class_name'], x_min, y_min, x_max, y_max]

    new_df.to_csv(src_csv.replace(".csv", "_resized.csv"), index=False)








if __name__ == '__main__':

    # src_dir = "/slurm_data/rawdata/ChestX-ray14/images/"
    # dest_dir = "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled_test/"


    # src_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs/"
    # dest_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs_rescaled/"
    # preprocess(src_dir, dest_dir)

    src_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs/"
    dest_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs_rescaled_320*320/"
    rescale_imgs(src_dir, dest_dir)

    # src_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs/"
    # dest_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs_rescaled_320*320/"
    # rescale_imgs(src_dir, dest_dir)


    # img_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs"
    # src_csv = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train.csv"
    # resize_bbox_annotation(img_dir, src_csv)

    # img_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs"
    # src_csv = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_test.csv"
    # resize_bbox_annotation(img_dir, src_csv)