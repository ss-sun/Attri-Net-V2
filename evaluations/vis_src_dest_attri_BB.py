import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import datetime


def draw_img_grid(src_folder, train_diseases, tgt_imgs):
    img_folder = os.path.join(src_folder, "attri-net_pixel_sensitivity_plots")
    img_list_dict={}
    for disease in train_diseases:
        image_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f)) and disease in f]
        img_list_dict[disease] = np.unique(np.array(image_files))
    draw_list = []
    if tgt_imgs is None:
        for disease in train_diseases:
            draw_list.append(img_list_dict[disease][:4])
    else:
        for disease in train_diseases:
            tgt_img = tgt_imgs[disease]
            tgt_list = [f for f in img_list_dict[disease] if tgt_img in f and disease in f]
            print(tgt_list)
            draw_list.append(tgt_list[:4])
            # draw_list = draw_list+ tgt_list

    num_diseases = len(train_diseases)
    fig, axs = plt.subplots(4, num_diseases, figsize=(3*num_diseases, 12))  # 3 rows, 5 columns

    # Row and column names
    row_names = ["attri", "counterfactual", "src", "srcBB"]
    col_names = train_diseases

    # Set row and column names as titles
    for i, row_name in enumerate(row_names):
        axs[i, 0].set_title(row_name)

    for j, col_name in enumerate(col_names):
        axs[0, j].set_title(col_name)

    # Iterate over image paths and plot each image
    for i in range(4):
        for j in range(num_diseases):
            # disease = train_diseases[j]
            # Calculate index in the image_paths list
            idx = i * num_diseases + j
            # Load image if available
            if idx < (4 * num_diseases):
                img_path = os.path.join(img_folder, draw_list[j][i])
                img = mpimg.imread(img_path)
                axs[i, j].imshow(img)

            axs[i, j].axis('off')  # Turn off axis labels and ticks

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the figure
    plt.show()



def draw_img_grid_otherXAI(src_folder, train_diseases, tgt_imgs):
    img_folder = os.path.join(src_folder, "bcos_resnet_pixel_sensitivity_plots")
    img_list_dict={}
    for disease in train_diseases:
        image_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f)) and disease in f]
        img_list_dict[disease] = np.unique(np.array(image_files))
    draw_list = []
    if tgt_imgs is None:
        for disease in train_diseases:
            draw_list.append(img_list_dict[disease][:4])
    else:
        for disease in train_diseases:
            tgt_img = tgt_imgs[disease]
            tgt_list = [f for f in img_list_dict[disease] if tgt_img in f and disease in f]
            print(tgt_list)
            draw_list.append(tgt_list[:2])
            # draw_list = draw_list+ tgt_list

    num_diseases = len(train_diseases)
    fig, axs = plt.subplots(2, num_diseases, figsize=(3*num_diseases, 6))  # 2 rows, num_diseases columns

    # Row and column names
    row_names = ["attri", "src"]
    col_names = train_diseases

    # Set row and column names as titles
    for i, row_name in enumerate(row_names):
        axs[i, 0].set_title(row_name)

    for j, col_name in enumerate(col_names):
        axs[0, j].set_title(col_name)

    # Iterate over image paths and plot each image
    for i in range(2):
        for j in range(num_diseases):
            # disease = train_diseases[j]
            # Calculate index in the image_paths list
            idx = i * num_diseases + j
            # Load image if available
            if idx < (2 * num_diseases):
                img_path = os.path.join(img_folder, draw_list[j][i])
                img = mpimg.imread(img_path)
                axs[i, j].imshow(img)

            axs[i, j].axis('off')  # Turn off axis labels and ticks

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the figure
    plt.show()





def draw_grid(img_list, src_folder, out_dir, prefix):
    current_time = datetime.datetime.now()
    prefix = str(current_time)[:-7]+prefix
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))  # 2 rows, 2 columns
    for i in range(2):
        for j in range(2):
            # disease = train_diseases[j]
            # Calculate index in the image_paths list
            idx = i * 2 + j
            # Load image if available
            if idx < 4:
                img_path = os.path.join(src_folder, img_list[idx])
                img = mpimg.imread(img_path)
                axs[i, j].imshow(img)
            axs[i, j].axis('off')  # Turn off axis labels and ticks
    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, prefix))
    plt.show()




def draw_2times2_img_grid(src_floder, disease, out_dir):
    draw_src_list = []
    draw_dest_list = []
    draw_attri_list = []
    for file in os.listdir(src_floder):
        if os.path.isfile(os.path.join(src_floder, file)):
            if disease in file and "rank0" in file:
                if "_src.jpg" in file:
                    draw_src_list.append(file)
                if "_dest.jpg" in file:
                    draw_dest_list.append(file)
                if "_attri.jpg" in file:
                    draw_attri_list.append(file)
    prefix = [disease + "_src" + ".jpg", disease + "_dest"+ ".jpg", disease + "_attri"+ ".jpg"]
    draw_lists = [draw_src_list, draw_dest_list, draw_attri_list]
    for i in range(len(draw_lists)):
        if len(draw_lists[i]) == 4:
            draw_grid(draw_lists[i], src_floder, out_dir, prefix=prefix[i])






if __name__ == '__main__':
    train_vindr_diseases=['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion']
    train_chexpert_diesease = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    train_nih_chestxray_disease = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
    out_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/results/plots"
    # model_dict = {
    #     "lambda30": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    #     "lambda40": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
    # }
    # train_diseases=['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion']
    # src_folder = os.path.join(model_dict["lambda40"], "pixel_sensitivity_result_dir", "attri-net")
    # draw_img_grid(src_folder, train_vindr_diseases)

    # model_dict = {
    #     "attri_nih": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:55--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    #     "bcos_nih": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:46-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    # }
    # # src_folder = os.path.join(model_dict["attri_nih"], "class_sensitivity_result_dir", "attri-net_class_sensitivity_plots")
    # src_folder = os.path.join(model_dict["bcos_nih"], "class_sensitivity_result_dir",
    #                           "bcos_resnet_class_sensitivity_plots")
    # print(src_folder)
    # draw_2times2_img_grid(src_folder, "Effusion", out_dir=out_dir)

    # model_dict = {
    #     "attri_nih": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:55--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    # }
    # src_folder = os.path.join(model_dict["attri_nih"],  "pixel_sensitivity_result_dir", "attri-net")
    # tgt_imgs = {
    #     "Atelectasis":"00029502_006",
    #     "Cardiomegaly":"00016564_000",
    #     "Effusion":"00008814_010",
    # }
    # draw_img_grid(src_folder, train_diseases=[train_nih_chestxray_disease[0], train_nih_chestxray_disease[1], train_nih_chestxray_disease[-1]], tgt_imgs=tgt_imgs)
    # draw_img_grid(src_folder, train_diseases=[train_nih_chestxray_disease[0], train_nih_chestxray_disease[1],
    #                                           train_nih_chestxray_disease[-1]], tgt_imgs=None)


    # model_dict = {
    #     "bcos_nih":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:46-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    #     "bcos_chexpert":"/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:25:38-chexpert-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    #     "bcos_vindr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet/bcos_resnet2023-11-11 20:38:25-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.1",
    #
    # }
    #
    # tgt_imgs_nih = {
    #     "Atelectasis":"00029502_006",
    #     "Cardiomegaly":"00016564_000",
    #     "Effusion":"00008814_010",
    # }
    #
    # tgt_imgs_chexpert = {
    #     "Atelectasis":"patient64821_study1_view1_frontal",
    #     "Cardiomegaly":"patient64749_study1_view1_frontal",
    #     "Consolidation":"patient64944_study1_view1_frontal",
    #     "Edema":"patient64743_study1_view1_frontal",
    #     "Pleural Effusion":"patient64833_study1_view1_frontal",
    # }
    #
    # tgt_imgs_vindr = {
    #     "Aortic enlargement": "0c6036df3708fe77c1c76498240d6774",
    #     "Cardiomegaly": "0aa034371e578904c6789c08e4118733",
    #     "Pulmonary fibrosis": "931dc77300612101a84eff6070ca270a",
    #     "Pleural thickening": "2a7272f9551b1b6248aba470735154aa",
    #     "Pleural effusion": "0a6fd1c1d71ff6f9e0f0afa746e223e4",
    # }
    # # src_folder = os.path.join(model_dict["bcos_nih"], "pixel_sensitivity_result_dir", "bcos_resnet")
    # # draw_img_grid_otherXAI(src_folder, train_diseases=[train_nih_chestxray_disease[0], train_nih_chestxray_disease[1],
    # #                                           train_nih_chestxray_disease[-1]], tgt_imgs=tgt_imgs_nih)
    #
    # src_folder = os.path.join(model_dict["bcos_vindr"], "pixel_sensitivity_result_dir", "bcos_resnet")
    # draw_img_grid_otherXAI(src_folder, train_diseases=train_vindr_diseases, tgt_imgs=tgt_imgs_vindr)

    # model_dict = {
    #     "attri_chexpert": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:03:38--chexpert--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
    # }
    # src_folder = os.path.join(model_dict["attri_chexpert"],  "pixel_sensitivity_result_dir", "attri-net")
    # # tgt_imgs = {
    # #     "Atelectasis":"patient64821_study1_view1_frontal",
    # #     "Cardiomegaly":"patient64749_study1_view1_frontal",
    # #     "Consolidation":"patient64944_study1_view1_frontal",
    # #     "Edema":"patient64743_study1_view1_frontal",
    # #     "Pleural Effusion":"patient64833_study1_view1_frontal",
    # # }
    # # draw_img_grid(src_folder, train_diseases=train_chexpert_diesease, tgt_imgs=tgt_imgs)
    # draw_img_grid(src_folder, train_diseases=train_chexpert_diesease, tgt_imgs=None)

    # model_dict = {
    #     "attri_vindr": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42"
    # }
    # src_folder = os.path.join(model_dict["attri_vindr"], "pixel_sensitivity_result_dir", "attri-net")
    # tgt_imgs = {
    #     "Aortic enlargement": "0c6036df3708fe77c1c76498240d6774",
    #     "Cardiomegaly": "0aa034371e578904c6789c08e4118733",
    #     "Pulmonary fibrosis": "931dc77300612101a84eff6070ca270a",
    #     "Pleural thickening": "2a7272f9551b1b6248aba470735154aa",
    #     "Pleural effusion": "0a6fd1c1d71ff6f9e0f0afa746e223e4",
    # }
    #draw_img_grid(src_folder, train_diseases=train_vindr_diseases, tgt_imgs=tgt_imgs)
    # draw_img_grid(src_folder, train_diseases=train_vindr_diseases, tgt_imgs=None)



    model_dict = {
        "attri_airgos": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-09 11:46:23--airogs_color--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=1--seed=42",
        "resnet_airgos": "/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet/resnet2023-11-14 09:42:54-airogs_color-bs=8-lr=0.0001-weight_decay=1e-05",
    }
    # src_folder = os.path.join(model_dict["attri_nih"], "class_sensitivity_result_dir", "attri-net_class_sensitivity_plots")
    src_folder = os.path.join(model_dict["attri_airgos"], "class_sensitivity_result_dir",
                              "attri-net_class_sensitivity_plots")

    # "attri-net_class_sensitivity_plots_weighted_map"
    print(src_folder)
    draw_2times2_img_grid(src_folder, "RG", out_dir=out_dir)