#!/home/gaojw/src/python3/python3/bin/python3
from model import proposed_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
import time
import random
import glob
import os
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Class names (make sure they are in the correct order)
class_names = ['NOR', 'LBBB', 'RBBB', 'APC', 'VFE']

imageh = 128
imagew = 128

inputH = 128
inputW = 192

# Build model and load trained weights
model = proposed_model()
lr = 0.0001
adm = Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
# model.summary()

# Load the pre-trained weights into the model
model.load_weights('result/fold_5_best_model.weights.h5')

# Load the test dataset
test_file = './MIT-BIH_AD_test.txt'
test_img_path = './MIT-BIH_AD/'
augmentation = True
output_img = False

# Open and shuffle test data
f = open(test_file, 'r')
lines = f.readlines()
random.shuffle(lines)

TP = 0
count = 0
total = len(lines)

# Initialize counters for each class
counter = {'NOR': 0, 'LBBB': 0, 'RBBB': 0, 'APC': 0, 'VFE': 0}
tp_counter = {'NOR': 0, 'LBBB': 0, 'RBBB': 0, 'APC': 0, 'VFE': 0}

# Iterate over each test sample
for line in tqdm(lines):
    path = line.split(' ')[0]
    label = line.split(' ')[-1]
    label = label.strip('\n')
    answer = int(label)

    img = os.path.join(test_img_path, path)

    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If augmentation is enabled, apply transformations
    if augmentation:
        Hshmean = int(np.round(np.max([0, np.round((imageh - inputH) / 2)])))
        Wshmean = int(np.round(np.max([0, np.round((imagew - inputW) / 2)])))
        image = image[Hshmean:Hshmean + inputH, Wshmean:Wshmean + inputW, :]
        image = cv2.resize(image, (imagew, imageh))
    
    # Prepare input data
    input_data = np.zeros((1, imagew, imageh, 3), dtype='float32')
    input_data = image.reshape(1, 128, 128, 3)

    # Make prediction
    pred = model.predict(input_data)
    label = np.argmax(pred)

    # Update TP counters and total counters
    if label == answer:
        TP += 1
        tp_counter[class_names[label]] += 1
    count += 1
    counter[class_names[answer]] += 1

# Print overall accuracy
print(f'Total: Acc = {TP / count}')

# Print accuracy per class
print(f'LBBB: {tp_counter["LBBB"]}/{counter["LBBB"]} = {tp_counter["LBBB"] / counter["LBBB"]}')
print(f'RBBB: {tp_counter["RBBB"]}/{counter["RBBB"]} = {tp_counter["RBBB"] / counter["RBBB"]}')
print(f'APC: {tp_counter["APC"]}/{counter["APC"]} = {tp_counter["APC"] / counter["APC"]}')
print(f'PVC: {tp_counter["VFE"]}/{counter["VFE"]} = {tp_counter["VFE"] / counter["VFE"]}')
print(f'NOR: {tp_counter["NOR"]}/{counter["NOR"]} = {tp_counter["NOR"] / counter["NOR"]}')

# Create a directory to store successfully recognized images
success_dir = '/home/deshill/yexijue/ECG_MIT_BIH_Arryhthmia_2D_CNN/success'
os.makedirs(success_dir, exist_ok=True)

# Save successfully recognized images into subdirectories
for class_name in class_names:
    class_dir = os.path.join(success_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

# Iterate over each test sample again to save successful predictions
for line in tqdm(lines):
    path = line.split(' ')[0]
    label = line.split(' ')[-1]
    label = label.strip('\n')
    answer = int(label)

    img = os.path.join(test_img_path, path)

    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if augmentation:
        Hshmean = int(np.round(np.max([0, np.round((imageh - inputH) / 2)])))
        Wshmean = int(np.round(np.max([0, np.round((imagew - inputW) / 2)])))
        image = image[Hshmean:Hshmean + inputH, Wshmean:Wshmean + inputW, :]
        image = cv2.resize(image, (imagew, imageh))

    input_data = np.zeros((1, imagew, imageh, 3), dtype='float32')
    input_data = image.reshape(1, 128, 128, 3)

    pred = model.predict(input_data)
    label = np.argmax(pred)

    if label == answer:
        # Save the image to the corresponding class folder
        class_folder = os.path.join(success_dir, class_names[label])
        save_path = os.path.join(class_folder, os.path.basename(img))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))