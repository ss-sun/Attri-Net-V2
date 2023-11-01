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

2. split the train set into train and validation set, keep the test set unchanged.