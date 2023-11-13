import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


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


if __name__ == '__main__':

    model_dict = {
        "lambda30": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 14:59:53--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=30.0--seed=42",
        "lambda40": "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net/attri-net2023-11-06 17:17:14--vindr_cxr_withBB--process_mask=previous--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01--l_loc=40.0--seed=42",
    }
    train_diseases=['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion']
    src_folder = os.path.join(model_dict["lambda40"], "pixel_sensitivity_result_dir", "attri-net")

    draw_img_grid(src_folder, train_diseases)



