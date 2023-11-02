Vindr-CXR dataset preprocessing

1. download dataset from physionet.org. since all chest x-ray are in .dicom format, we need to convert them to png images.

run following script to convert .dicom to png images.

python3 ./data/preprocess_vindr.py \
  --input-dir "/mnt/qb/rawdata/vindr-cxr-physionet-original/1.0.0/train" \
  --output-dir "/mnt/qb/rawdata/vindr-cxr-physionet-pngs/1.0.0/train" \
  --cpus 1 \
  --log-file "/mnt/qb/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_train_log.txt" \
  --out-file "/mnt/qb/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_train_results.csv" \


python3 ./data/preprocess_vindr.py \
  --input-dir "/mnt/qb/rawdata/vindr-cxr-physionet-original/1.0.0/test" \
  --output-dir "/mnt/qb/rawdata/vindr-cxr-physionet-pngs/1.0.0/test" \
  --cpus 1 \
  --log-file "/mnt/qb/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_test_log.txt" \
  --out-file "/mnt/qb/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_test_results.csv" \


2. to speed up train, I resized all images in vindr to 320*320, and also scale the boundbox annotation. using scripts in preprocess_nih_vindr_datasets.py
Now the dataset document as following:
"train_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs_rescaled_320*320",
"test_image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/test_pngs_rescaled_320*320",
"train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_train_resized.csv",
"test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/annotations/annotations_test_resized.csv",


3. In the dataset module, I split the train set into train and validation set, keep the test set unchanged.



airogs dataset preprocessing
1. download
2. resize using script preprocess_airogs.py
3. 