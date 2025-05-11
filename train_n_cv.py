from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from model import proposed_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

tf.config.optimizer.set_experimental_options({'disable_xla': False})

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def plot_history(history, result_dir, fold):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title(f'model accuracy - fold {fold}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, f'fold_{fold}_model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title(f'model loss - fold {fold}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, f'fold_{fold}_model_loss.png'))
    plt.close()

def save_history(history, result_dir, fold):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, f'fold_{fold}_result.txt'), 'w') as fp:
        fp.write('epoch\\tloss\\tacc\\tval_loss\\tval_acc\\n')
        for i in range(nb_epoch):
            fp.write('{}\\t{}\\t{}\\t{}\\t{}\\n'.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

def data_generator(lines, img_path, inputH, inputW, batch_size, train=True):
    num_samples = len(lines)
    while True:
        random.shuffle(lines)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_lines = lines[start:end]
            batch, labels = process_batch(batch_lines, img_path, inputH, inputW, train)
            labels = to_categorical(labels, num_classes=5)  # 确保标签是 one-hot 编码
            yield batch, labels

def process_batch(lines, img_path, inputH, inputW, train=True):
    num = len(lines)
    batch = np.zeros((num, inputH, inputW, 3), dtype='float32')
    labels = np.zeros(num, dtype='int')

    for i in range(num):
        path = lines[i].split(' ')[0]
        label = int(lines[i].split(' ')[-1].strip())

        img = os.path.join(img_path, path)
        image = cv2.imread(img)
        if image is None:
            raise ValueError(f"Image not found or corrupted: {img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if train:
            crop_x = random.randint(0, max(0, image.shape[1] - inputW))
            image = image[:, crop_x:crop_x + inputW, :]
        else:
            center_x = image.shape[1] // 2
            half_w = inputW // 2
            start_x = max(0, center_x - half_w)
            end_x = start_x + inputW
            if end_x > image.shape[1]:
                start_x = image.shape[1] - inputW
                end_x = image.shape[1]
            image = image[:, start_x:end_x, :]

        image = cv2.resize(image, (inputW, inputH))

        batch[i] = image
        labels[i] = label

    return batch, labels

def cross_validate_model(model, train_file, num_classes, img_path, inputH, inputW, epochs, batch_size, n_splits=5):
    with open(train_file, 'r') as f:
        lines = f.readlines()
    num_samples = len(lines)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 1
    history_list = []
    best_model_weights = None
    best_val_acc = 0

    for train_idx, val_idx in skf.split(np.zeros(num_samples), np.array([int(line.split()[-1].strip()) for line in lines])):
        print(f"Training fold {fold}/{n_splits}")

        train_lines = [lines[i] for i in train_idx]
        val_lines = [lines[i] for i in val_idx]

        train_generator = data_generator(train_lines, img_path, inputH, inputW, batch_size, train=True)
        val_generator = data_generator(val_lines, img_path, inputH, inputW, batch_size, train=False)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # 添加EarlyStopping回调
        early_stopping = EarlyStopping(
            monitor='val_loss',  # 监控验证集损失
            patience=10,         # 允许验证损失不改善的轮数
            verbose=1,           # 输出日志
            restore_best_weights=True  # 恢复最佳权重
        )

        checkpoint = ModelCheckpoint(f'./result/fold_{fold}_best_model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_lines) // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_lines) // batch_size,
            callbacks=[checkpoint, early_stopping],  # 添加早停回调
            verbose=1
        )

        model.load_weights(f'./result/fold_{fold}_best_model.weights.h5')

        val_loss, val_acc = model.evaluate(val_generator, steps=len(val_lines) // batch_size, verbose=0)
        print(f"Validation Accuracy for fold {fold}: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.get_weights()

        history_list.append(history)

        fold += 1

    model.set_weights(best_model_weights)

    return model, history_list

n_splits = 5
batch_size = 32
epochs = 100
input_h = 128
input_w = 128
train_file = 'MIT-BIH_AD_train.txt'
test_file = 'MIT-BIH_AD_val.txt'
num_classes = 5
img_path = './MIT-BIH_AD/'

model = proposed_model(nb_classes=num_classes)

lr = 0.0001
adam = Adam(learning_rate=lr)

model, history_list = cross_validate_model(
    model, 
    train_file, 
    num_classes, 
    img_path, 
    input_h, 
    input_w, 
    epochs, 
    batch_size, 
    n_splits=n_splits
)

for i, history in enumerate(history_list):
    plot_history(history, f'./result', i+1)

save_history(history_list[-1], './result', 'final')
