import os

src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net"

def filt_files(src_dir, keywords):
    # Get a list of all filenames in the directory
    filenames = [f for f in os.listdir(src_dir)]
    # Filter filenames by keywords
    filtered_filenames = [filename for filename in filenames if all(keyword in filename for keyword in keywords)]
    pathes = [os.path.join(src_dir, filename) for filename in filtered_filenames]
    return pathes


if __name__ == '__main__':
    src_dir = "/mnt/qb/work/baumgartner/sun22/TMI_exps/attri-net"
    keywords = ["attri-net2023-11-05", "chexpert", "--l_loc="]
    pathes = filt_files(src_dir, keywords)
    for path in pathes:
        print(path)


