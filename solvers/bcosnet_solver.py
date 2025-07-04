from __future__ import print_function
import os
import torch
import torch.nn as nn
from models.bcos_net import resnet50 as bcos_resnet50
import numpy as np
import wandb
from train_utils import to_numpy
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from captum.attr import InputXGradient
import json
from models.losses import EnergyPointingGameBBMultipleLoss_multilabel, PseudoEnergyLoss_multilabel
from data.pseudo_guidance_dict import pseudo_mask_dict, weighted_pseudo_mask_dict


class bcos_resnet_solver(object):
    # Train and test ResNet50.
    def __init__(self, exp_configs, data_loader):

        # self.debug = exp_configs.debug
        self.print_loss = False
        self.exp_configs = exp_configs
        self.TRAIN_DISEASES = exp_configs.train_diseases
        self.use_gpu = exp_configs.use_gpu
        self.use_wandb = exp_configs.use_wandb
        self.dataloaders = data_loader


        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if exp_configs.mode == "train":
            self.debug = exp_configs.debug
            self.dataset = exp_configs.dataset
            self.lambda_loc = exp_configs.lambda_localizationloss
            self.guidance_mode = exp_configs.guidance_mode
            self.img_size = exp_configs.image_size
            self.epochs = exp_configs.epochs
            self.lr = exp_configs.lr
            self.weight_decay = exp_configs.weight_decay
            self.ckpt_dir = exp_configs.ckpt_dir
            self.train_loader = data_loader['train']
            self.valid_loader = data_loader['valid']
            self.test_loader = data_loader['test']

            # if self.guidance_mode == "pseudo_mask":
            #     self.pseudoMask = self.prepare_pseudoMask(exp_configs.dataset)
            #     self.local_loss = PseudoEnergyLoss_multilabel()

            if self.guidance_mode == "full_guidance":
                self.local_loss_gt = EnergyPointingGameBBMultipleLoss_multilabel()

            if self.guidance_mode in["mixed", "mixed_weighted"]:
                self.train_with_few_bbox = True
                self.dloader_bbox = data_loader['train_pos_bbox']
                self.bbox_data_iters = iter(self.dloader_bbox)
                if self.guidance_mode == "mixed":
                    self.pseudoMask = self.prepare_pseudoMask(exp_configs.dataset)
                if self.guidance_mode == "mixed_weighted":
                    self.pseudoMask = self.prepare_weighted_pseudoMask(exp_configs.dataset)
                self.local_loss_gt = EnergyPointingGameBBMultipleLoss_multilabel()
                self.local_loss_pseudo = PseudoEnergyLoss_multilabel()


            # Initialize model.
            self.model = self.init_model()
            self.loss = torch.nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if exp_configs.mode == "test":
            self.model_path = exp_configs.model_path
            self.valid_loader = data_loader['valid']
            self.test_loader = data_loader['test']
            self.model = self.init_model()
            self.load_model(self.model_path)

    def prepare_pseudoMask(self, dataset):
        pseudoMask = {}
        file_path = pseudo_mask_dict[dataset]
        with open(file_path) as json_file:
            data = json.load(json_file)
            for disease in self.TRAIN_DISEASES:
                pseudoMask[disease] = np.array(data[disease])
        return pseudoMask


    def prepare_weighted_pseudoMask(self, dataset):
        weighted_pseudoMask = {}
        file_path = weighted_pseudo_mask_dict[dataset]
        with open(file_path) as json_file:
            data = json.load(json_file)
            for disease in self.TRAIN_DISEASES:
                weighted_pseudoMask[disease] = np.array(data[disease])
        return weighted_pseudoMask


    def extend_channels(self, img):
        return torch.cat((img, 1-img), dim=1)

    def init_model(self):
        # Prepare model, input dim=1 because input x-ray images are gray scale.
        model = bcos_resnet50(pretrained=False, num_classes=len(self.TRAIN_DISEASES), in_chans=2)
        model = model.to(self.device)
        return model

    def load_model(self, model_path):
        model_dir = model_path + "/ckpt"
        model_path = os.path.join(model_dir, 'best_classifier.pth')
        self.model.load_state_dict(torch.load(model_path))

    def get_tgt_bbox(self, labels, bboxs):
        tgt_bbox = []
        if self.dataset in ["vindr_cxr", "vindr_cxr_mix"]: # only vindr_cxr has multiple bboxs
            for idx in range(len(labels)):
                if labels[idx] == 1:
                    tgt_bbox.append(bboxs[idx])
        else:
            tgt_bbox = bboxs.unsqueeze(0) # 'chexpert_mix' and 'nih_chestxray' only have one bbox or mask for multilabel.
        return tgt_bbox

    def get_pseudo_mask(self, labels):
        pseudo_mask = []
        for idx in range(len(labels)):
            if labels[idx] == 1:
                disease = self.TRAIN_DISEASES[idx]
                pseudo_mask.append(self.pseudoMask[disease])
        return pseudo_mask

    def get_gt_batch(self):
        try:
            batch = next(self.bbox_data_iters)
        except StopIteration:
            self.bbox_data_iters = iter(self.dloader_bbox)
            batch = next(self.bbox_data_iters)
        return batch



    def train(self):
        max_batches = len(self.valid_loader)
        best_valid_auc = 0.0
        self.model.train()
        for epoch in range(self.epochs):
            if self.debug:
                print("Epoch: {}".format(epoch))
            steps = 0
            train_epoch_loss = 0.0
            for idx, data in enumerate(self.train_loader):
                self.gt_batch = False
                if self.debug:
                    print("steps: ", steps)
                    if steps==50:
                        self.validation(max_batches=5)

                train_data = data['img']
                # attri = self.get_attributes(train_data[0].unsqueeze(0), label_idx=1, positive_only=True)
                train_data = self.extend_channels(train_data)
                train_labels = data['label']
                train_data = train_data.to(self.device)
                train_labels = train_labels.to(self.device)
                y_pred = self.model(train_data)
                cls_loss = self.loss(y_pred, train_labels)

                # compute localization loss
                localization_loss = 0
                if self.guidance_mode == "full_guidance" and torch.sum(train_labels) > 0: # only for vindr-cxr dataset which all samples have bbox
                    local_loss_list = []
                    for img_idx in range(len(train_data)):
                        img = train_data[img_idx]
                        lbl = train_labels[img_idx]
                        if torch.sum(lbl) > 0:
                            attributions = self.get_attribution_list(img.unsqueeze(0), lbl, positive_only=True)
                            tgt_bboxs = self.get_tgt_bbox(train_labels[img_idx], data['BBox'][img_idx])
                            local_loss_list.append(self.local_loss_gt(attributions, tgt_bboxs))
                    if len(local_loss_list) > 0:
                        localization_loss = torch.mean(torch.stack(local_loss_list))* self.lambda_loc


                if self.guidance_mode == "pseudo_mask"and torch.sum(train_labels) > 0:
                    local_loss_list = []
                    for img_idx in range(len(train_data)):
                        img = train_data[img_idx]
                        lbl = train_labels[img_idx]
                        if torch.sum(lbl) > 0:
                            attributions = self.get_attribution_list(img.unsqueeze(0), lbl, positive_only=True)
                            pseudo_masks = self.get_pseudo_mask(train_labels[img_idx])
                            local_loss_list.append(self.local_loss_pseudo(attributions, pseudo_masks))
                    if len(local_loss_list) > 0:
                        localization_loss = torch.mean(torch.stack(local_loss_list))* self.lambda_loc

                if self.guidance_mode in ["mixed", "mixed_weighted"] and torch.sum(train_labels) > 0:
                    if idx % 10 == 0:
                        # for every 10 batches, get one batch with gt annotations. otherswise use pseudo annotations.
                        print("idx: ", idx)
                        data = self.get_gt_batch()

                        train_data = data['img']
                        train_data = self.extend_channels(train_data)
                        train_labels = data['label']
                        train_data = train_data.to(self.device)
                        train_labels = train_labels.to(self.device)
                        y_pred = self.model(train_data)
                        cls_loss = self.loss(y_pred, train_labels)
                        train_bbox = data['BBox']

                        # batch with gt annotations
                        local_loss_list = []
                        for img_idx in range(len(train_data)):
                            img = train_data[img_idx]
                            lbl = train_labels[img_idx]
                            if torch.sum(lbl) > 0:
                                attributions = self.get_attribution_list(img.unsqueeze(0), lbl, positive_only=True)
                                tgt_bboxs = self.get_tgt_bbox(train_labels[img_idx], train_bbox[img_idx])
                                if self.dataset =="chexpert_mix":
                                    local_loss_list.append(self.local_loss_gt(attributions, tgt_bboxs, isbbox=False))
                                else:
                                    local_loss_list.append(self.local_loss_gt(attributions, tgt_bboxs, isbbox=True))
                        if len(local_loss_list) > 0:
                            localization_loss = torch.mean(torch.stack(local_loss_list))* self.lambda_loc
                    else:
                        # batch without gt annotation, use pseudo guidance
                        local_loss_list = []
                        for img_idx in range(len(train_data)):
                            img = train_data[img_idx]
                            lbl = train_labels[img_idx]
                            if torch.sum(lbl) > 0:
                                attributions = self.get_attribution_list(img.unsqueeze(0), lbl, positive_only=True)
                                pseudo_masks = self.get_pseudo_mask(train_labels[img_idx])
                                local_loss_list.append(self.local_loss_pseudo(attributions, pseudo_masks))
                        if len(local_loss_list) > 0:
                            localization_loss = torch.mean(torch.stack(local_loss_list))* self.lambda_loc



                train_loss= cls_loss + localization_loss

                train_epoch_loss += train_loss.item()

                if self.use_wandb:
                    wandb.log({"train_step_cls_loss": cls_loss})
                    wandb.log({"train_step_cls_loss": localization_loss})
                elif self.print_loss:
                    print("train_step_cls_loss: ", cls_loss)
                    print("train_step_localization_loss: ", localization_loss)
                    print("train_loss: ", train_loss)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                steps += 1
            train_epoch_loss = train_epoch_loss / steps
            if self.use_wandb:
                wandb.log({"train_epoch_loss": train_epoch_loss})
            else:
                print("train_epoch_loss: ", train_epoch_loss)



            # Validation
            valid_auc_mean, _, _ = self.validation(max_batches=max_batches)
            # Save best model
            if best_valid_auc <= valid_auc_mean:
                best_valid_auc = valid_auc_mean
                if self.use_wandb:
                    wandb.log({"best_valid_auc": best_valid_auc})
                else:
                    print("best_valid_auc: ", best_valid_auc)
                torch.save(self.model.state_dict(), '{:}/best_classifier.pth'.format(self.ckpt_dir))
            print('Epoch=%s, valid_AUC=%.4f, Best_valid_AUC=%.4f' % (epoch, valid_auc_mean, best_valid_auc))
            self.model.train()


    def validation(self, max_batches):
        self.model.eval()
        with torch.no_grad():
            valid_pred = []
            valid_true = []
            valid_epoch_loss = 0.0
            steps = 0

            for jdx, data in enumerate(self.valid_loader):
                if jdx >= max_batches:
                    break
                valid_data = data['img']
                valid_data = self.extend_channels(valid_data)
                valid_labels = data['label']
                valid_data = valid_data.to(self.device)
                valid_labels = valid_labels.to(self.device)
                y_pred_logits = self.model(valid_data)
                y_pred = torch.sigmoid(y_pred_logits)
                valid_loss = self.loss(y_pred_logits, valid_labels)
                valid_epoch_loss += valid_loss.item()
                if self.use_wandb:
                    wandb.log({"valid_step_loss": valid_loss})
                valid_pred.append(to_numpy(y_pred))
                valid_true.append(to_numpy(valid_labels))
                steps += 1

            valid_epoch_loss = valid_epoch_loss / steps
            if self.use_wandb:
                wandb.log({"valid_epoch_loss": valid_epoch_loss})
            else:
                print("valid_epoch_loss: ", valid_epoch_loss)

            valid_true = np.concatenate(valid_true)
            valid_pred = np.concatenate(valid_pred)
            valid_auc_mean = roc_auc_score(valid_true, valid_pred)
            if self.use_wandb:
                wandb.log({"valid_auc_mean": valid_auc_mean})
            else:
                print("valid_auc_mean: ", valid_auc_mean)

            for i in range(len(self.TRAIN_DISEASES)):
                t = valid_true[:, i]
                p = valid_pred[:, i]
                auc = roc_auc_score(t, p)
                print(self.TRAIN_DISEASES[i] + ": " + str(auc))
                if self.use_wandb:
                    wandb.log({self.TRAIN_DISEASES[i]: auc})
                else:
                    print(self.TRAIN_DISEASES[i] + "auc: " + str(auc))

        return valid_auc_mean, valid_true, valid_pred


    def test(self, which_loader="test", save_result=False, result_dir=None):

        if which_loader == "test":
            data_loader = self.test_loader
        if which_loader == "valid":
            data_loader = self.valid_loader

        self.model.eval()
        with torch.no_grad():
            test_pred = []
            test_true = []
            diter = iter(data_loader)
            for i in tqdm(range(len(data_loader))):
                data = next(diter)
                test_data = data['img']
                test_data = self.extend_channels(test_data)
                test_labels = data['label']
                test_data = test_data.to(self.device)
                test_labels = test_labels.to(self.device)
                y_pred_logits = self.model(test_data)
                y_pred = torch.sigmoid(y_pred_logits)
                test_pred.append(to_numpy(y_pred))
                test_true.append(to_numpy(test_labels))

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_auc_mean = roc_auc_score(test_true, test_pred)
            print('test_auc_mean: ', test_auc_mean)

            for i in range(len(self.TRAIN_DISEASES)):
                t = test_true[:, i]
                p = test_pred[:, i]
                auc = roc_auc_score(t, p)
                print(self.TRAIN_DISEASES[i] + " auc: " + str(auc))

            pred = np.asarray(test_pred)
            true = np.asarray(test_true)

            if save_result:
                pred_path = os.path.join(result_dir, which_loader + '_pred.txt')
                true_path = os.path.join(result_dir, which_loader + '_true.txt')
                np.savetxt(pred_path, pred)
                np.savetxt(true_path, true)

        return test_pred, test_true, test_auc_mean




    def get_optimal_thresholds(self, save_result=True, result_dir=None):

        pred, true, valid_auc = self.test(which_loader="valid", save_result=True, result_dir=result_dir)
        print("validation auc: ", valid_auc)
        best_threshold = {}
        threshold = []

        for i in range(len(self.TRAIN_DISEASES)):
            disease = self.TRAIN_DISEASES[i]
            statics = {}
            p = pred[:, i]
            t = true[:, i]
            statics['fpr'], statics['tpr'], statics['threshold'] = roc_curve(t, p, pos_label=1)
            sensitivity = statics['tpr']
            specificity = 1 - statics['fpr']
            sum = sensitivity + specificity  # this is Youden index
            best_t = statics['threshold'][np.argmax(sum)]
            best_threshold[disease] = best_t
            threshold.append(best_t)
        threshold = np.asarray(threshold)
        print(best_threshold)

        if save_result:
            path = os.path.join(result_dir, 'best_threshold.txt')
            np.savetxt(path, threshold)
        return best_threshold




    def get_attribution_list(self, img, labels, positive_only=True):
        explainer = InputXGradient(self.model)
        attribution_list = []
        for idx in range(len(labels)):
            if labels[idx] == 1:
                input = torch.autograd.Variable(img.clone().detach(), requires_grad=True)
                attr_map = explainer.attribute(input, target=idx)[:, 0, :, :]
                if positive_only:
                    attr_map = torch.relu(attr_map)
                attribution_list.append(attr_map)
        return attribution_list





    def get_attributes(self, img, label_idx, positive_only=True):
        """
        Get the attributions of the image for the given label index.
        """
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        img = self.extend_channels(img)
        img = img.to(self.device)
        self.explainer = InputXGradient(self.model)
        input = torch.autograd.Variable(img, requires_grad=True)
        attr_maps = self.explainer.attribute(input, target=label_idx)[:, 0, :, :]
        if positive_only:
            attr_maps = torch.relu(attr_maps)
        # attr_maps = self.explainer.get_attributions(input=img, target_label_idx=label_idx, positive_only=positive_only)
        return attr_maps

    def get_probs(self, inputs, label_idx):
        """
        Get the probability of the given input image with respect to a specific disease.
        """
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        img = self.extend_channels(inputs)
        img = img.to(self.device)
        y_pred_logits = self.model(img)
        y_pred = torch.sigmoid(y_pred_logits)
        prob = y_pred[:, label_idx]
        return prob
