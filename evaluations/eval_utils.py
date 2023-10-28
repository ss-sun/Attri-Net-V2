import os
import numpy as np
from train_utils import to_numpy
from PIL import Image, ImageDraw


# def reshape_weights(weights_1d):
#     len = max(weights_1d.shape)
#     len_sqrt = int(np.sqrt(len))
#     weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
#     return weights_2d
#
# def upsample_weights(weights_2d, target_size):
#     source_size = weights_2d.shape[0]
#     up_ratio = int(target_size / source_size)
#     upsampled_weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))
#     return upsampled_weights


def get_weighted_map(map, lgs):
    map = to_numpy(map)
    weights_1d = to_numpy(lgs.linear.weight.data)
    # target_size = map.shape[0]
    len = max(weights_1d.shape)
    len_sqrt = int(np.sqrt(len))
    weights_2d = weights_1d.reshape((len_sqrt, len_sqrt))
    source_size = weights_2d.shape[0]
    up_ratio = int(map.shape[0] / source_size)
    weights = np.kron(weights_2d, np.ones((up_ratio, up_ratio)))
    weightedmap = np.multiply(map, weights)
    # the weighted map will be compare with bounding box, so we need to get the biggest positive value.
    # we want to get the element that has the biggest influence, therefore we choose highest absolute value.
    # abs_weightedmap = np.absolute(to_numpy(weightedmap.squeeze()))

    return weightedmap


def draw_BB(img, box, img_id, prefix, out_dir):

    x_min = box[0]
    y_min = box[1]
    width = box[2]
    height = box[3]

    rgb_img = Image.fromarray(img * 255).convert('RGB')
    draw = ImageDraw.Draw(rgb_img)
    outline_color = (255, 0, 0)
    outline_width = 2
    draw.rectangle([x_min, y_min, x_min+width, y_min+height], outline=outline_color, width=outline_width)
    del draw
    path = os.path.join(out_dir, img_id + '_' + prefix + '_with_BB.png')
    rgb_img.save(path)
    return path



def draw_hit(pos, attr_map, gt_annotation, prefix, output_dir, gt="bbox"):
    # attr_map = to_numpy(attr_map * 0.5 + 0.5).squeeze()
    attr_img = Image.fromarray((attr_map * 255).astype(np.uint8)).convert('RGB')

    # Set the size of the red dot
    size = 5
    # Calculate the coordinates of the circle
    x0 = pos[1] - size
    y0 = pos[0] - size
    x1 = pos[1] + size
    y1 = pos[0] + size

    if gt == "bbox":
        bbox = gt_annotation
        x_min = int(bbox[0].item())
        y_min = int(bbox[1].item())
        x_max = x_min + int(bbox[2].item())
        y_max = y_min + int(bbox[3].item())

        draw = ImageDraw.Draw(attr_img)
        draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
        draw.rectangle((x_min, y_min, x_max, y_max), fill=None, outline=(0, 255, 0))
        attr_img.save(os.path.join(output_dir, prefix + '.png'))

    if gt == "mask":
        gt_mask = gt_annotation
        rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = gt_mask * 255
        rgb_mask[:, :, 2] = attr_map * 255
        img = Image.fromarray(rgb_mask).convert('RGB')
        # Create a new ImageDraw object
        draw = ImageDraw.Draw(img)
        img.putalpha(50)
        # Draw the circle
        draw.ellipse((x0, y0, x1, y1), fill=(255, 0, 0))
        attr_img.paste(img, (0, 0), img)
        # src_img.show()
        attr_img.save(os.path.join(output_dir, prefix + '.jpg'))
