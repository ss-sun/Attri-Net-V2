from PIL import Image
import os
import shutil
import tqdm as tqdm

# This file scales images in airogs fundus dataset to speed up training. original image have large size e.g 1956*1934.
# we scale the image to the size of 512, i.e the minimum dimension of the image is 512. And keep the original aspect ratio.
# Images from these two datasets have large dimensions that affect the training speed if we scale during training.

def preprocess(src_dir, dest_dir, basesize=336):
    # Clear dest_dir
    try:
        shutil.rmtree(dest_dir)
    except:
        pass
    os.makedirs(dest_dir)
    file_lists = os.listdir(src_dir)
    for file in file_lists:
        if file.endswith(('.jpg', '.png', 'jpeg')):
            print(file)
            src_img_path = os.path.join(src_dir, file)
            img = Image.open(src_img_path)
            (width, height) = img.size[-2:]
            if height >= width:
                new_width = basesize
                new_height = int(height * (basesize / width))
            else:
                new_width = int(width * (basesize / height))
                new_height = basesize
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img.save(os.path.join(dest_dir, file))


if __name__ == '__main__':
    src_dir = "/mnt/qb/work/baumgartner/sun22/data/5793241/train"
    dest_dir = "/mnt/qb/work/baumgartner/sun22/data/5793241/train_scaled_336"
    preprocess(src_dir, dest_dir)