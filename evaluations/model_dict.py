
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


resnet_model_path_dict = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-07-28 13:27:38-chexpert--official_datasplit-orientation=Frontal-augmentation=previous-bs=4-lr=0.0001-weight_decay=1e-05",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:22-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:34-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05",
    "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-09-22 16:20:52-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
}

bcos_resnet_model_path_dict = {
    "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:18:35-chexpert-bs=8-lr=0.0001-weight_decay=1e-05",
    "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:19:08-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
    "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-02 20:17:29-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05",
    "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-09-22 16:24:51-skmtea-bs=8-lr=0.0001-weight_decay=1e-05",
}

attrinet_model_path_dict = {
        "chexpert_previous": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        # 0.299 "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:42:26--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=color_jitter--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        # 0.331 "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:03--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=all--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        # 0.176 center crop "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:42:00--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=center_crop--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        # 0.3831 previous "chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-08-01 18:43:37--chexpert--official_datasplit-orientation=Frontal-image_size=320-augmentation=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 18:10:38--vindr_cxr--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "skmtea": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-09-22 13:19:07--skmtea--bs=4--lg_ds=32--l_cri=1.0--l1=500.0--l2=1000.0--l3=500.0--l_ctr=0.01",
        "chexpert_abs(mx)":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-28 19:51:00--chexpert--process_mask=abs(mx)--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
        "chexpert_sum(abs(mx))":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-10-29 17:13:06--chexpert--process_mask=sum(abs(mx))--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
        "airogs":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-02 14:56:36--airogs--process_mask=previous--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
        "vindr_cxr_previous_1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:50:41--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1.0--seed=42",
        "vindr_cxr_previous_5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:50:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=5.0--seed=42",
        "vindr_cxr_previous_10":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:50:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=10.0--seed=42",
        "vindr_cxr_previous_20":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:51:47--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=20.0--seed=42",
        "vindr_cxr_previous_100":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-03 12:51:51--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=100.0--seed=42",
        "airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:46:23--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1--seed=42",

}  # new models


# attrinet_with_psydomask_dict_unfinished={
#     "lambda1": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:01:21--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1.0--seed=42",
#     "lambda5": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:03:09--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=5.0--seed=42",
#     "lambda10": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:04:01--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=10.0--seed=42",
#     "lambda20": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:04:59--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=20.0--seed=42",
#     "lambda100": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-05 14:05:01--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=100.0--seed=42",
# }

attrinet_vindrBB_different_lambda_dict={
    "lambda25": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:00:22--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
    "lambda30": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    "lambda40": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
    "lambda50": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:00:51--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=50.0--seed=42",
    "lambda60": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:01:24--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=60.0--seed=42",
    "lambda75": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:01:44--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=75.0--seed=42",
    "lambda80": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=80.0--seed=42",
}


glaucoma_dict={
    "vagan_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:48:19--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=0.0--l_ctr=0.0--seed=42",
    "attrinet_airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:46:23--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1--seed=42",
    "bcos_resnet_airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-13 21:13:08-airogs_color-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1", # model not train well
    "resnet_airogs_color": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-11-14 09:42:54-airogs_color-bs=8-lr=0.0001-weight_decay=1e-05",
}


# bcos_vindr_with_guidance_dict={
#     "lambda1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1",
#     "lambda0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.5",
#     "lambda0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
#     "lambda0.05":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:18-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05",
# }





bcos_chexpert_with_guidance_dict={
    "lambda1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:41-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1",
    "lambda0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:37-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.5",
    "lambda0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:38-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    "lambda0.05":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:39-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05",
}

bcos_nih_chestxray_with_guidance_dict={
    "lambda1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:41-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=1",
    "lambda0.5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:40-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.5",
    "lambda0.1":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:46-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    "lambda0.05":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:41-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05",
}


# attrinet_vindr_cxr_withBB_with_guidance_dict={
#     "lambda25": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 15:00:22--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
#     "lambda30": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
#     "lambda40": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
# }


attrinet_chexpert_with_guidance_dict={
    "lambda25":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
    "lambda30":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    "lambda35":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=35.0--seed=42",
    "lambda40":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
}

attrinet_nih_chestxray_with_guidance_dict={
    "lambda25":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:58--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25.0--seed=42",
    "lambda30":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:55--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    "lambda35":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:46--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=35.0--seed=42",
    "lambda40":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:44--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
}


bcos_vindr_with_guidance_dict={
    "lambda0.02":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:26-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.02",
    "lambda0.01":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.01",
    "lambda0.0075":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:24-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0075",
    "lambda0.005":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.005",
    "lambda0.0025":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.0025",
    "lambda0":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-17 17:33:32-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0",
}

attrinet_vindr_cxr_withBB_with_guidance_dict={
    "lambda20":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=20.0--seed=42",
    "lambda15":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=15.0--seed=42",
    "lambda10":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=10.0--seed=42",
    "lambda5":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=5.0--seed=42",
    "lambda0":"/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-17 18:31:11--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--seed=42",
}

# attrinet_nih_withBB_with_guidance_dict={
#     "lambda25": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-24 17:45:52--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=25--seed=42",
# }

attrinet_nih_withBB_with_guidance_dict={
    "lambda0": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
}