import torch
from models.attrinet_modules import Discriminator_with_Ada, Generator_with_Ada, CenterLoss
from models.lgs_classifier import LogisticRegressionModel
from models.bcos_net import resnet50 as bcos_resnet50








def get_attrinet_model_size():
    """
    Get the size of the Attrinet model components.
    """

    num_class = 5
    img_size = 320
    n_ones = 20
    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


    device = "cuda" if torch.cuda.is_available() else "cpu"


    net_g = Generator_with_Ada(num_classes=num_class,
                                    img_size=img_size,
                                    act_func="relu", n_fc=8,
                                    dim_latent=num_class * n_ones,
                                    conv_dim=64, in_channels=1, out_channels=1,
                                    repeat_num=6)
    net_d = Discriminator_with_Ada(act_func="relu", in_channels=1,
                                        conv_dim=64,
                                        dim_latent=num_class * n_ones,
                                        repeat_num=6)

    net_g.to(device)
    net_d.to(device)

    # Initialize one classifier for each disease.
    net_lgs = {}
    for disease in TRAIN_DISEASES:
        m = LogisticRegressionModel(
            input_size=img_size, num_classes=1,
            downsample_ratio=32)
        m.to(device)
        net_lgs[disease] = m


    feat_dim = img_size * img_size
    center_losses = {}
    optimizer_centloss = {}
    for disease in TRAIN_DISEASES:
        loss = CenterLoss(num_classes=2, feat_dim=feat_dim, device=device)
        opt = torch.optim.SGD(loss.parameters(), lr=0.1)
        center_losses[disease] = loss
        optimizer_centloss[disease] = opt




    size_g = sum(p.numel() for p in net_g.parameters() if p.requires_grad)
    size_d = sum(p.numel() for p in net_d.parameters() if p.requires_grad)
    size_lgs=0
    for disease, model in net_lgs.items():
        size_lgs += sum(p.numel() for p in model.parameters() if p.requires_grad)


    size_ctr = 0
    for disease, loss in center_losses.items():
        size_ctr += sum(p.numel() for p in loss.parameters() if p.requires_grad)




    print("size of Generator with Ada: ", size_g)
    print("size of Discriminator with Ada: ", size_d)
    print("size of Logistic Regression Classifier: ", size_lgs)
    print("size of Center Loss: ", size_ctr)
    print("Total size of model: ", size_g + size_d + size_lgs + size_ctr)


def get_bcos_model_size():

    TRAIN_DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                      "Pleural Effusion"]

    model = bcos_resnet50(pretrained=False,
                          num_classes=len(TRAIN_DISEASES), in_chans=2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters in BCOS model: ", total_params)




if __name__ == "__main__":
    model_name = "bcos"
    if model_name == "attrinet":
        get_attrinet_model_size()

    elif model_name == "bcos":
        get_bcos_model_size()
