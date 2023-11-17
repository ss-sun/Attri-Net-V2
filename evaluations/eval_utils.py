import os
import numpy as np
from train_utils import to_numpy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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



def create_mask_fromBB(img_size, bbox):
    #bbox: [x, y, w, h]
    mask = np.zeros(img_size)
    mask[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
    return mask



def vis_samples_withMask(src_img, attr, dests, gt_annotation, prefix, output_dir, attr_method):
    gt_mask = gt_annotation
    src_img = to_numpy(src_img * 0.5 + 0.5).squeeze()
    src_img = Image.fromarray(src_img * 255).convert('RGB')
    rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[:, :, 1] = gt_mask * 255
    mask_img = Image.fromarray(rgb_mask).convert('RGB')
    mask_img.putalpha(50)
    src_img.paste(mask_img, (0, 0), mask_img)
    # src_img.show()
    src_img.save(os.path.join(output_dir, prefix + '_src.jpg'))

    if attr_method == "attri-net": # for attributions from attri-net, draw both attribution maps and the destination counterfactual images
        # here we assume the attributions are weighted maps.so we don't need to flip the attributions.
        attr = to_numpy(attr * 0.5 + 0.5).squeeze()
        attri_img = plt.cm.bwr(attr)  # use bwr color map, here negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
        attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
        attri_img.paste(mask_img, (0, 0), mask_img)
        # attri_img.show()
        attri_img.save(os.path.join(output_dir, prefix + '_attri.jpg'))

        dest_img = to_numpy(dests * 0.5 + 0.5).squeeze()
        dest_img = Image.fromarray(dest_img * 255).convert('RGB')
        rgb_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[:, :, 1] = gt_mask * 255
        mask_img = Image.fromarray(rgb_mask).convert('RGB')
        mask_img.putalpha(50)
        dest_img.paste(mask_img, (0, 0), mask_img)
        dest_img.save(os.path.join(output_dir, prefix + '_dest.jpg'))

    else:
        attr = to_numpy(attr).squeeze()  # for other attributions, we keep all positive attributions.
        vmax = np.max(attr)
        scaled_attr = attr / vmax # convert to (0,1)
        attri_img = plt.cm.bwr(scaled_attr * 0.5+0.5)  # use bwr color map, we want negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
        attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
        attri_img.paste(mask_img, (0, 0), mask_img)
        # attri_img.show()
        attri_img.save(os.path.join(output_dir, prefix + '_attri.jpg'))



def vis_samples(src_img, attr, dests, prefix, output_dir, attr_method):

    src_img = to_numpy(src_img * 0.5 + 0.5).squeeze()
    if len(src_img.shape) == 2:
        src_img = Image.fromarray(src_img * 255).convert('RGB')
    else:
        src_img = (src_img * 255).astype(np.uint8)
        src_img = src_img.transpose((1, 2, 0))
        src_img = Image.fromarray(src_img)
    # src_img.show()
    src_img.save(os.path.join(output_dir, prefix + '_src.jpg'))

    if attr_method == "attri-net": # for attributions from attri-net, draw both attribution maps and the destination counterfactual images
        # here we assume the attributions are weighted maps.so we don't need to flip the attributions.
        attr = to_numpy(attr * 0.5 + 0.5).squeeze()
        attri_img = plt.cm.bwr(attr)  # use bwr color map, here negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
        attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
        # attri_img.show()
        attri_img.save(os.path.join(output_dir, prefix + '_attri.jpg'))

        dest_img = to_numpy(dests * 0.5 + 0.5).squeeze()
        if len(dest_img.shape) == 2:
            dest_img = Image.fromarray(dest_img * 255).convert('RGB')
        else:
            dest_img = (dest_img * 255).astype(np.uint8)
            dest_img = dest_img.transpose((1, 2, 0))
            dest_img = Image.fromarray(dest_img)
        # dest_img.show()
        dest_img.save(os.path.join(output_dir, prefix + '_dest.jpg'))

    else:
        attr = to_numpy(attr).squeeze()  # for other attributions, we keep all positive attributions.
        vmax = np.max(attr)
        scaled_attr = attr / vmax # convert to (0,1)
        if len(scaled_attr.shape) == 3:
            scaled_attr = np.mean(scaled_attr, axis=0).squeeze()
        attri_img = plt.cm.bwr(scaled_attr * 0.5+0.5)  # use bwr color map, we want negative values are blue, positive values are red, 0 is white. need to convert to value (0,1), negative values corrsponding to (0-0.5), positive to (0.5,1), white=0.5
        attri_img = Image.fromarray((attri_img * 255).astype(np.uint8)).convert('RGB')
        # attri_img.show()
        attri_img.save(os.path.join(output_dir, prefix + '_attri.jpg'))








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


