import sys
import os
sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/project/tmi"))
import numpy as np
import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, accuracy_score
from scipy.stats import mode

def str2bool(v):
    return v.lower() in ('true')

class classification_analyser():
    """
    Analyse the classification performance of the model.
    """
    def __init__(self, solver, config):
        self.solver = solver
        self.result_dir = config.result_dir
        self.best_threshold = {}
        os.makedirs(self.result_dir, exist_ok=True)

    # def run(self):
    #     _, _, auc = self.solver.test()
    #     return auc

    def run(self):
        # Read data from file.
        threshold_path = os.path.join(self.result_dir, "best_threshold.txt")
        if os.path.exists(threshold_path):
            print("threshold alreay computed, load threshold")
            threshold = np.loadtxt(open(threshold_path))
            for c in range(len(self.solver.TRAIN_DISEASES)):
                disease = self.solver.TRAIN_DISEASES[c]
                self.best_threshold[disease] = threshold[c]
                print(self.best_threshold[disease])

        else:
            self.best_threshold = self.solver.get_optimal_thresholds(save_result=True, result_dir=self.result_dir)
        pred_file = os.path.join(self.result_dir, "test_pred.txt")

        if os.path.exists(pred_file):
            print("prediction of test set already made")
            self.test_pred = np.loadtxt(os.path.join(self.result_dir, "test_pred.txt"))
            self.test_true = np.loadtxt(os.path.join(self.result_dir, "test_true.txt"))
        else:
            self.test_pred, self.test_true, class_auc = self.solver.test(which_loader="test", save_result=True,
                                                                         result_dir=self.result_dir)
            print('test_class_auc', class_auc)

def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='attri-net', choices=['resnet', 'attri-net', 'bcos_resnet'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    return parser


def get_arguments(model_path):
    parser = argument_parser()
    opts = parser.parse_args()
    opts.model_path = model_path
    print("evaluate attri-net model: " + opts.model_path)
    # Configurations of networks
    opts.image_size = 320
    opts.n_fc = 8
    opts.n_ones = 20
    opts.num_out_channels = 1
    opts.lgs_downsample_ratio = 32
    opts.result_dir = os.path.join(opts.model_path, "result_dir")
    return opts


def prep_solver(datamodule, exp_configs):
    data_loader = {}
    if "attri-net" in exp_configs.exp_name:
        train_loaders = datamodule.single_disease_train_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)
        vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=exp_configs.batch_size, shuffle=False)
        valid_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader['train_pos'] = train_loaders['pos']
        data_loader['train_neg'] = train_loaders['neg']
        data_loader['vis_pos'] = vis_dataloaders['pos']
        data_loader['vis_neg'] = vis_dataloaders['neg']
        data_loader['valid'] = valid_loader
        data_loader['test'] = test_loader
        solver = task_switch_solver(exp_configs, data_loader=data_loader)
    return solver



def main():
    from data.dataset_params import dataset_dict, data_default_params
    attrinet_model_path_dict = {
        "chexpert1": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "chexpert2":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-28 14:35:06--chexpert--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=12345",
        "chexpert3":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-28 14:36:51--chexpert--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=2023",
        "chexpert4":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-28 14:37:24--chexpert--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=8675309",
        "chexpert5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-28 14:37:54--chexpert--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=21",
    }
    train_disease = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    # for key, value in attrinet_model_path_dict.items():
    #     print("now evaluating model: ", key)
    #     model_path = value
    #     config = get_arguments(model_path)
    #     datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    #     solver = prep_solver(datamodule, config)
    #     analyser = classification_analyser(solver, config)
    #     analyser.run()
    #

    threds = {}
    preds = {}
    gts = {}
    results = []
    for key, value in attrinet_model_path_dict.items():
        result_dir = os.path.join(value, "result_dir")
        thred_file = np.loadtxt(os.path.join(result_dir, "best_threshold.txt"))
        pred_file = np.loadtxt(os.path.join(result_dir, "test_pred.txt"))
        true_file = np.loadtxt(os.path.join(result_dir, "test_true.txt"))
        threds[key] = thred_file
        preds[key] = pred_file
        gts[key] = true_file
        results.append([pred_file])

    avg_predict = np.mean(np.array(results), axis=0).squeeze()
    gt = gts['chexpert1']
    auc_mean = roc_auc_score(gt, avg_predict)
    print("auc_mean: ",auc_mean)


    predictions = []
    accu = []
    for key, value in preds.items():
        pred_file = value
        thred_file = threds[key]
        gt =gts[key]
        pred = np.zeros(pred_file.shape)
        for i in range(len(train_disease)):
            pred[np.where(pred_file[:, i] >= thred_file[i])[0], i] = 1
            pred[np.where(pred_file[:, i] < thred_file[i])[0], i] = 0
        predictions.append(pred)
        accuracy = np.mean(gt == pred)
        print("accuracy: ", accuracy)
        same_values = (gt == pred)
        # Count the number of True values in the result
        num_same_values = np.sum(same_values)
        print("num_same_values",num_same_values)
        print("ratio: ", num_same_values / gt.size)
        accu.append(accuracy)

    print("average accuracy: ", np.mean(np.array(accu)))

    gt = gts['chexpert1']
    stacked_array = np.stack(predictions)

    # Compute the majority votes along the new dimension (axis=0)
    majority_votes, _ = mode(stacked_array, axis=0)
    majority_vote_acc = np.mean(gt == majority_votes)
    print(majority_vote_acc)












if __name__ == "__main__":
    main()