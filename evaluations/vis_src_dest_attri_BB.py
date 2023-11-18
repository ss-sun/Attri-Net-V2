import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import datetime


def draw_img_grid(src_folder, train_diseases):
    img_folder = os.path.join(src_folder, "attri-net_pixel_sensitivity_plots")
    img_list_dict={}
    for disease in train_diseases:
        image_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f)) and disease in f]
        img_list_dict[disease] = np.unique(np.array(image_files))
    draw_list = []
    for disease in train_diseases:
        draw_list.append(img_list_dict[disease][:3])

    fig, axs = plt.subplots(3, 5, figsize=(15, 9))  # 3 rows, 5 columns

    # Row and column names
    row_names = ["attri", "dest", "src"]
    col_names = train_diseases

    # Set row and column names as titles
    for i, row_name in enumerate(row_names):
        axs[i, 0].set_title(row_name)

    for j, col_name in enumerate(col_names):
        axs[0, j].set_title(col_name)

    # Iterate over image paths and plot each image
    for i in range(3):
        for j in range(5):
            # disease = train_diseases[j]
            # Calculate index in the image_paths list
            idx = i * 5 + j
            # Load image if available
            if idx < 15:

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
    model_dict = {
        "nih": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-10 21:04:55--nih_chestxray--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42"
    }
    src_folder = os.path.join(model_dict["nih"], "class_sensitivity_result_dir", "attri-net_class_sensitivity_plots")
    print(src_folder)
    draw_2times2_img_grid(src_folder, "Effusion", out_dir=out_dir)








