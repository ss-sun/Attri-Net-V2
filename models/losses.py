import torch

"""
Source: https://github.com/sukrutrao/Model-Guidance/blob/main/losses.py
"""

def get_localization_loss(loss_name):
    loss_map = {
        "Energy": EnergyPointingGameBBMultipleLoss,
        "L1": GradiaBBMultipleLoss,
        "RRR": RRRBBMultipleLoss,
        "PPCE": HAICSBBMultipleLoss
    }
    return loss_map[loss_name]()


class BBMultipleLoss:

    def __init__(self):
        super().__init__()

    def __call__(self, attributions, bb_coordinates):
        raise NotImplementedError

    def get_bb_mask(self, bb_coordinates, mask_shape):
        bb_mask = torch.zeros(mask_shape, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        return bb_mask


class EnergyPointingGameBBMultipleLoss_multilabel:

    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, bb_coordinates):
        loss_list = []
        for i in range(len(bb_coordinates)):
            pos_attribution = attributions[i].clamp(min=0) # original code
            bb_mask = torch.zeros_like(pos_attribution, dtype=torch.long)
            coord = bb_coordinates[i]
            xmin, ymin, width, height = coord
            bb_mask[:, int(ymin):int(ymin+height), int(xmin):int(xmin+width)] = 1
            num = pos_attribution[torch.where(bb_mask == 1)].sum()
            den = pos_attribution.sum()
            if den < 1e-7:
                loss = 1-num
            else:
                loss = 1-num/den
            loss_list.append(loss)
        return sum(loss_list)/len(loss_list)


class PseudoEnergyLoss_multilabel:

    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, psydo_mask):
        """
        Compute localization loss with psydo mask.
        """
        loss_list = []
        for i in range(len(psydo_mask)):
            pos_attribution = attributions[i].clamp(min=0)  # original code
            mask = torch.from_numpy(psydo_mask[i]).unsqueeze(0)
            num = pos_attribution[torch.where(mask == 1)].sum()
            den = pos_attribution.sum()
            if den < 1e-7:
                loss = 1 - num
            else:
                loss = 1 - num / den
            loss_list.append(loss)
        return sum(loss_list) / len(loss_list)







class EnergyPointingGameBBMultipleLoss:

    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, bb_coordinates):
        pos_attributions = torch.abs(attributions)  # modified code, adapted to attri-net.
        bb_mask = torch.zeros_like(pos_attributions, dtype=torch.long)
        xmin, ymin, width, height = bb_coordinates.numpy().astype(int)  # modified code, adapted to attri-net.
        xmax = xmin + width
        ymax = ymin + height
        bb_mask[:,ymin:ymax, xmin:xmax] = 1
        num = pos_attributions[torch.where(bb_mask == 1)].sum()
        den = pos_attributions.sum()
        if den < 1e-7:
            return 1 - num
        return 1 - num / den

class PseudoEnergyLoss:
    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, psydo_mask):
        """
        Compute localization loss with psydo mask.
        """
        pos_attributions = torch.abs(attributions)  # modified code, adapted to attri-net.
        psydo_mask = torch.from_numpy(psydo_mask).unsqueeze(0).cuda()
        num = pos_attributions[torch.where(psydo_mask == 1)].sum()
        den = pos_attributions.sum()
        if den < 1e-7:
            return 1-num

        return 1-num/den






class RRRBBMultipleLoss(BBMultipleLoss):

    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape)
        irrelevant_attrs = attributions[torch.where(bb_mask == 0)]
        return torch.square(irrelevant_attrs).sum()


class GradiaBBMultipleLoss(BBMultipleLoss):

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.only_positive = True
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape).cuda()
        return self.l1_loss(attributions, bb_mask)


class HAICSBBMultipleLoss(BBMultipleLoss):

    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.only_positive = True
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape)
        attributions_in_box = attributions[torch.where(bb_mask == 1)]
        return self.bce_loss(attributions_in_box, torch.ones_like(attributions_in_box))