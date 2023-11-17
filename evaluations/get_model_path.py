import os
import json

src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net"

def filt_files(src_dir, keywords):
    # Get a list of all filenames in the directory
    filenames = [f for f in os.listdir(src_dir)]
    # Filter filenames by keywords
    filtered_filenames = [filename for filename in filenames if all(keyword in filename for keyword in keywords)]
    pathes = [os.path.join(src_dir, filename) for filename in filtered_filenames]
    return pathes


if __name__ == '__main__':
    # src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net"
    # keywords = ["attri-net2023-11-06", "vindr_cxr_withBB", "--l_loc="]

    # src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/bcos_resnet"
    # keywords = ["bcos_resnet2023-11-11", "20:25", "nih", "lambda_localizationloss="]

    # src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net"
    # keywords = ["attri-net2023-11-10", "21:03", "chexpert", "--l_loc="]

    src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net"
    keywords = ["attri-net2023-11-10", "21:04:", "nih_chestxray", "--l_loc="]


    pathes = filt_files(src_dir, keywords)
    for path in pathes:
        print(path)



# "bcos_resnet2023-11-11 20:38:18-vindr_cxr_withBB-bs=8-lr=0.0001-weight_decay=1e-05-lambda_localizationloss=0.05"



    # results_dict = {}
    # for i in range(1, 11):
    #     key = "lambda_" + str(i)
    #     values = {"t1": 1, "t2": 2}
    #     results_dict[key] = values
    #



