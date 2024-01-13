resnet_models = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-12-07 10:14:57-chexpert-bs=4-lr=0.0001-weight_decay=1e-05",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-12-07 10:03:30-nih_chestxray-bs=4-lr=0.0001-weight_decay=1e-05",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-12-07 14:42:26-vindr_cxr-bs=4-lr=0.0001-weight_decay=1e-05"
    }


bcos_resnet_models = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-12-07 16:44:27-chexpert-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-12-07 16:20:17-nih_chestxray-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-12-07 16:44:27-vindr_cxr-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0",
}

attrinet_models = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 17:29:36--chexpert--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 17:29:36--nih_chestxray--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 17:30:35--vindr_cxr--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
}

guided_bcos_resnet_models = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-12-22 20:51:01-chexpert_mix-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2024-01-05 17:31:39-nih_chestxray-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-12-22 20:54:08-vindr_cxr_mix-bs=4-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
}

guided_attrinet_models = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:48:15--chexpert_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-11 15:40:33--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 18:51:38--vindr_cxr_mix--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42",
    "vindr_cxr_full_super": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 18:12:17--vindr_cxr--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--full_guidance--l_loc=30.0--guid_freq=0.0--seed=42",
}






aba_loss_attrinet_models = {
    "cls":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 18:11:38--nih_chestxray--l_cri=0.0--l1=0.0--l2=0.0--l_cls=100.0--l_ctr=0.0--no_guidance--seed=42",
    "cls_adv":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 18:14:42--nih_chestxray--l_cri=1.0--l1=0.0--l2=0.0--l_cls=100.0--l_ctr=0.0--no_guidance--seed=42",
    "cls_adv_reg":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 18:14:46--nih_chestxray--l_cri=1.0--l1=100.0--l2=200.0--l_cls=100.0--l_ctr=0.0--no_guidance--seed=42",
    # below is the same model as in dictionary attrinet_models, to keep the results in ablation study consistent with the main paper
    "cls_adv_reg_ctr":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-08 17:29:36--nih_chestxray--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--no_guidance--seed=42",
}

# aba_guidance_attrinet_models = {
#     "mixed_weighted_mask":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:55:43--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed_weighted--l_loc=30.0--guid_freq=0.1--seed=42",
# }


aba_guidance_attrinet_models = {
    "bbox":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-11 14:22:48--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--bbox/masks--l_loc=30.0--guid_freq=0.1--seed=42",
    "pseudo_mask":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-11 14:29:11--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--pseudo_mask--l_loc=30.0--guid_freq=0.0--seed=42",
    "pseudo_bbox":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-12 15:34:15--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--pseudo_bbox--l_loc=30.0--guid_freq=0.0--seed=42",
    "mixed":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-11 15:40:33--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed--l_loc=30.0--guid_freq=0.1--seed=42",
    "mixed_weighted_mask":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-12-21 16:55:43--nih_chestxray--l_cri=1.0--l1=100--l2=200--l_cls=100--l_ctr=0.01--mixed_weighted--l_loc=30.0--guid_freq=0.1--seed=42",
}












# resnet_models = {
#     "chexpert": "/mnt/qb/work/baumgartner/sun22/MT_exps/resnet_cls2022-12-08 14:30:54-dataset=chexpert-bs=4-lr=0.0001-weight_decay=1e-05",
#     "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/MT_exps/resnet_cls2022-12-08 10:24:56-dataset=nih_chestxray-bs=4-lr=0.0001-weight_decay=1e-05",
#     "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/MT_exps/resnet_cls2022-12-07 18:44:06-dataset=vindr_cxr-bs=4-lr=0.0001-weight_decay=1e-05"
#     }

# attrinet_models = {
#     "chexpert": "/mnt/qb/work/baumgartner/sun22/MT_exps/ours_1209/ours2022-12-10 13:16:32-dataset=chexpert-batch_size=4-logreg_dsratio=32-G_loss_type=with_center_loss-lambda_1=100-lambda_2=200-lambda_3=100-centerloss_lambda=0.01",
#     "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/MT_exps/ours_1209/ours2022-12-10 13:14:34-dataset=nih_chestxray-batch_size=4-logreg_dsratio=32-G_loss_type=with_center_loss-lambda_1=100-lambda_2=200-lambda_3=100-centerloss_lambda=0.01",
#     "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/MT_exps/ours_1209/ours2022-12-10 13:13:32-dataset=vindr_cxr-batch_size=4-logreg_dsratio=32-G_loss_type=with_center_loss-lambda_1=100-lambda_2=200-lambda_3=100-centerloss_lambda=0.01"
# }

#
# resnet_model_path_dict = {
#     "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-07-28 13:27:38-chexpert--official_datasplit-orientation=Frontal-augmentation=previous-bs=4-lr=0.0001-weight_decay=1e-05",
#     "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:22-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
#     "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:34-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05",
#     "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-09-22 16:20:52-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
# }
#
# bcos_resnet_model_path_dict = {
#     "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:18:35-chexpert-bs=8-lr=0.0001-weight_decay=1e-05",
#     "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:19:08-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
#     "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:17:29-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05",
#     "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-22 16:24:51-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
# }
#
# attrinet_model_path_dict = {
#         "chexpert_previous": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         # 0.299 "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:42:26--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=color_jitter--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         # 0.331 "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:03--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=all--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         # 0.176 center crop "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:42:00--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=center_crop--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         # 0.3831 previous "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 18:10:38--vindr_cxr--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
#         "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-22 13:19:07--skmtea--bs=4--lg_ds=32--l_cri=1.0--l1=500.0--l2=1000.0--l3=500.0--l_ctr=0.01",
#         "chexpert_abs(mx)":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-28 19:51:00--chexpert--process_mask=abs(mx)--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
#         "chexpert_sum(abs(mx))":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-29 17:13:06--chexpert--process_mask=sum(abs(mx))--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
#         "airogs":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-02 14:56:36--airogs--process_mask=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
#         "vindr_cxr_previous_1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:50:41--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1.0--seed=42",
#         "vindr_cxr_previous_5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:50:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=5.0--seed=42",
#         "vindr_cxr_previous_10":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:50:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=10.0--seed=42",
#         "vindr_cxr_previous_20":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:51:47--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=20.0--seed=42",
#         "vindr_cxr_previous_100":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:51:51--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=100.0--seed=42",
#         "airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:46:23--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1--seed=42",
#
# }  # new models
#
#
# # attrinet_with_psydomask_dict_unfinished={
# #     "lambda1": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:01:21--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1.0--seed=42",
# #     "lambda5": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:03:09--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=5.0--seed=42",
# #     "lambda10": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:04:01--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=10.0--seed=42",
# #     "lambda20": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:04:59--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=20.0--seed=42",
# #     "lambda100": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:05:01--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=100.0--seed=42",
# # }
#
# attrinet_vindrBB_different_lambda_dict={
#     "lambda25": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:00:22--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
#     "lambda30": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
#     "lambda40": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
#     "lambda50": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:00:51--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=50.0--seed=42",
#     "lambda60": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:01:24--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=60.0--seed=42",
#     "lambda75": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:01:44--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=75.0--seed=42",
#     "lambda80": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=80.0--seed=42",
# }
#
#
# glaucoma_dict={
#     "vagan_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:48:19--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=0.0--l_ctr=0.0--seed=42",
#     "attrinet_airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:46:23--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1--seed=42",
#     "bcos_resnet_airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-13 21:13:08-airogs_color-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1", # model not train well
#     "resnet_airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-11-14 09:42:54-airogs_color-bs=8-lr=0.0001-weight_decay=1e-05",
# }
#
#
# # bcos_vindr_with_guidance_dict={
# #     "lambda1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1",
# #     "lambda0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.5",
# #     "lambda0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
# #     "lambda0.05":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:18-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05",
# # }
#
#
#
#
#
# bcos_chexpert_with_guidance_dict={
#     "lambda1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:41-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1",
#     "lambda0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:37-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.5",
#     "lambda0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:38-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
#     "lambda0.05":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:39-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05",
# }
#
# bcos_nih_chestxray_with_guidance_dict={
#     "lambda1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:41-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1",
#     "lambda0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:40-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.5",
#     "lambda0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:46-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
#     "lambda0.05":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:41-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05",
# }
#
#
# # attrinet_vindr_cxr_withBB_with_guidance_dict={
# #     "lambda25": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:00:22--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
# #     "lambda30": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
# #     "lambda40": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
# # }
#
#
# attrinet_chexpert_with_guidance_dict={
#     "lambda25":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
#     "lambda30":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
#     "lambda35":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=35.0--seed=42",
#     "lambda40":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
# }
#
# attrinet_nih_chestxray_with_guidance_dict={
#     "lambda25":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:58--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
#     "lambda30":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:55--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
#     "lambda35":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:46--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=35.0--seed=42",
#     "lambda40":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:44--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
# }
#
#
# bcos_vindr_with_guidance_dict={
#     "lambda0.02":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:26-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.02",
#     "lambda0.01":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.01",
#     "lambda0.0075":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:24-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0075",
#     "lambda0.005":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.005",
#     "lambda0.0025":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0025",
#     "lambda0":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0",
# }
#
# attrinet_vindr_cxr_withBB_with_guidance_dict={
#     "lambda20":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=20.0--seed=42",
#     "lambda15":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=15.0--seed=42",
#     "lambda10":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=10.0--seed=42",
#     "lambda5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=5.0--seed=42",
#     "lambda0":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
# }
#
# # attrinet_nih_withBB_with_guidance_dict={
# #     "lambda25": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-24 17:45:52--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25--seed=42",
# # }
#
# attrinet_nih_withBB_with_guidance_dict={
#     "lambda0": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
# }
#
# attrinet_nih_withBB_with_guidance_different_freq_dict={
#     "fre0.005":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-28 22:53:24--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30--guid_freq=0.005--seed=42",
#     "fre0.05": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-28 22:52:58--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30--guid_freq=0.05--seed=42",
#     "fre0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-28 18:17:37--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30--guid_freq=0.1--seed=42",
#     "fre0.2":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-28 18:29:15--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30--guid_freq=0.2--seed=42",
#     "fre0.4":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-28 18:51:12--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30--guid_freq=0.4--seed=42",
#     "fre0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-28 22:09:01--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30--guid_freq=0.5--seed=42",
# }