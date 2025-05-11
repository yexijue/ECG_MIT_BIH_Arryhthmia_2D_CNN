# ECG_MIT_BIH_Arryhthmia_2D_CNN
Use 2D-CNN to deal with mit_bih_arryhthmia dataset (python) 
## enviroment
python 3.11
pip install -r requirements.txt

## step 1 
download the mit-bih-arryhthmia dataset and unzip it (you may get a fileholder named mit-bih-arrhythmia-database-1.0.0) （Check get_data.txt）
## step 2
python data_preprocess.py
## step 3
python split_train_test_dataset.py
## step 4
python train.py / train_n_cv.py
## step 5
python predict.py

## Notice
- Enhanced data
- Classification Pipeline (With Cross-Validation)

