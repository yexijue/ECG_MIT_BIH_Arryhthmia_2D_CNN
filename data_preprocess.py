from glob import glob
import wfdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn import preprocessing

def get_records():
    paths = glob('/home/deshill/yexijue/ECG_MIT_BIH_Arryhthmia_2D_CNN/data/physionet.org/files/mitdb/1.0.0/*.atr')
    paths = [path[:-4] for path in paths]
    paths.sort()
    return paths

def segmentation(records, type, output_dir=''):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    kernel = np.ones((4, 4), np.uint8)
    count = 1

    for e in tqdm(records, desc=f"Processing type {type}"):
        signals, _ = wfdb.rdsamp(e, channels=[0])
        signals = preprocessing.scale(np.nan_to_num(signals))
        ann = wfdb.rdann(e, 'atr')

        ids = np.isin(ann.symbol, [type])
        imp_beats = ann.sample[ids]
        beats = list(ann.sample)

        for i in imp_beats:
            j = beats.index(i)
            if j != 0 and j != len(beats) - 1 and beats[j] - 96 >= 0 and beats[j] + 96 < len(signals):
                data = signals[beats[j]-96: beats[j]+96, 0]
                plt.plot(data, linewidth=0.5)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                filename = os.path.join(output_dir, f'fig_{count}.png')
                plt.savefig(filename)
                plt.close()

                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.erode(im_gray, kernel, iterations=1)
                im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(filename, im_gray)
                results.append(filename)

                count += 1

    return results

def augment_image(image_path, save_dir, count):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    max_shift = 10
    direction = random.choice(['up', 'down', 'left', 'right'])
    shift = random.randint(1, max_shift)

    if direction == 'up':
        cropped = img[shift:, :]
        padded = cv2.copyMakeBorder(cropped, shift, 0, 0, 0, cv2.BORDER_REPLICATE)
    elif direction == 'down':
        cropped = img[:-shift, :]
        padded = cv2.copyMakeBorder(cropped, 0, shift, 0, 0, cv2.BORDER_REPLICATE)
    elif direction == 'left':
        cropped = img[:, shift:]
        padded = cv2.copyMakeBorder(cropped, 0, 0, shift, 0, cv2.BORDER_REPLICATE)
    elif direction == 'right':
        cropped = img[:, :-shift]
        padded = cv2.copyMakeBorder(cropped, 0, 0, 0, shift, cv2.BORDER_REPLICATE)

    padded = cv2.resize(padded, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    out_path = os.path.join(save_dir, f"aug_shift_{count}.png")
    cv2.imwrite(out_path, padded)
    return [out_path]

def balance_samples(pathes_by_type, output_dirs, labels, augment=True, target_samples=8000):
    balanced_paths = {}

    for type, paths in pathes_by_type.items():
        original_paths = paths.copy()
        current_count = len(paths)
        save_dir = f"./MIT-BIH_AD/{output_dirs[labels.index(type)]}"

        print(f"[{type}] Original: {current_count} samples")

        # 保留已有大于等于目标数量的类别
        if current_count >= target_samples:
            balanced_paths[type] = paths
            print(f"→ No augmentation needed for '{type}'.")
            continue

        # 增强不足的类别
        count = 0
        while len(paths) < target_samples:
            sample_path = random.choice(original_paths)
            if augment:
                aug_paths = augment_image(sample_path, save_dir, count)
                paths.extend(aug_paths)
            else:
                paths.append(sample_path)
            count += 1

        balanced_paths[type] = paths[:target_samples]
        print(f"→ Augmented to {len(paths[:target_samples])} samples for '{type}'.")

    return balanced_paths

def write_dataset_split(balanced_paths, labels, output_dirs):
    train_list, val_list, test_list = [], [], []

    for type, paths in balanced_paths.items():
        random.shuffle(paths)
        num_train = int(len(paths) * 0.6)
        num_val = int(len(paths) * 0.2)

        train_list.extend([(p, type) for p in paths[:num_train]])
        val_list.extend([(p, type) for p in paths[num_train:num_train + num_val]])
        test_list.extend([(p, type) for p in paths[num_train + num_val:]])

    def write_to_file(filename, data_list):
        with open(filename, 'w') as f:
            for path, label in data_list:
                f.write(f"{path} {label}\n")

    write_to_file('MIT-BIH_AD_train.txt', train_list)
    write_to_file('MIT-BIH_AD_val.txt', val_list)
    write_to_file('MIT-BIH_AD_test.txt', test_list)

    print(f"\n[Dataset Split]")
    print(f"Training samples: {len(train_list)}")
    print(f"Validation samples: {len(val_list)}")
    print(f"Test samples: {len(test_list)}")
    print(f"Total samples: {len(train_list) + len(val_list) + len(test_list)}")

if __name__ == "__main__":
    records = get_records()

    # 正常心拍，左束支传导阻滞，右束支传导阻滞，房性早搏，心室颤动
    labels = ['N', 'L', 'R', 'A', '!']
    output_dirs = ['NOR/', 'LBBB/', 'RBBB/', 'APC/', 'VFE/']

    pathes_by_type = {}
    for label, out_dir in zip(labels, output_dirs):
        paths = segmentation(records, label, output_dir=f'./MIT-BIH_AD/{out_dir}')
        pathes_by_type[label] = paths

    # 平衡样本数至 target_samples（如 8000）
    balanced_paths = balance_samples(pathes_by_type, output_dirs, labels, augment=True, target_samples=8000)

    # 划分训练/验证/测试集
    write_dataset_split(balanced_paths, labels, output_dirs)

