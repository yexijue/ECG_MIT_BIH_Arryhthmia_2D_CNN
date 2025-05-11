from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import random
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import glob
import math
from collections import Counter
import tensorflow as tf
from callbacks import Step
from model import proposed_model
import warnings
warnings.filterwarnings("ignore")
# 禁用或启用 XLA
tf.config.optimizer.set_experimental_options({'disable_xla': False})

# Ensure that GPUs are available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# # 查看可用的 GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 设置 GPU 内存动态增长（防止占用全部 GPU 内存）
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"Using GPU: {gpus}")
#     except RuntimeError as e:
#         print(f"Error setting memory growth: {e}")
# else:
#     print("No GPU available, using CPU.")

# Plot training history
def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

# Save history in a text file
def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
            i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

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

        # 裁剪并统一尺寸
        if train:
            crop_x = random.randint(0, max(0, image.shape[1] - inputW))
            image = image[:, crop_x:crop_x + inputW, :]
        else:
            center_x = image.shape[1] // 2
            half_w = inputW // 2
            start_x = max(0, center_x - half_w)
            end_x = start_x + inputW
            if end_x > image.shape[1]:  # 若越界，则反向截取
                start_x = image.shape[1] - inputW
                end_x = image.shape[1]
            image = image[:, start_x:end_x, :]

        image = cv2.resize(image, (inputW, inputH))  # 最终确保尺寸一致

        batch[i] = image
        labels[i] = label

    return batch, labels


# Generator function for training batch
def generator_train_batch( train_txt, batch_size, num_classes, img_path, inputH, inputW ):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)     # Total number of images
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])  # Shuffle the lines

        for i in range(int(num/batch_size)):  # Generate batches
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch(new_line[a:b], img_path, inputH, inputW, train=True)
            y = to_categorical(np.array(x_labels), num_classes)  # One-hot encoding
            yield x_train, y  # Generate batches

# # Generator function for validation batch
# def generator_val_batch(val_txt,batch_size,num_classes,img_path,inputH,inputW):
#     f = open(val_txt, 'r')
#     lines = f.readlines()
#     num = len(lines)
#     while True:
#         new_line = []
#         index = [n for n in range(num)]
#         random.shuffle(index)
#         for m in range(num):
#             new_line.append(lines[index[m]])
#         for i in range(int(num / batch_size)):
#             a = i * batch_size
#             b = (i + 1) * batch_size
#             y_test,y_labels = process_batch(new_line[a:b],img_path,inputH,inputW,train=False)
#             y = to_categorical(np.array(y_labels), num_classes)
#             yield y_test, y
# 替换为
# Load full validation set once to avoid generator randomness
def load_val_data(val_txt, img_path, inputH, inputW, num_classes):
    with open(val_txt, 'r') as f:
        lines = f.readlines()
    x_val, y_labels = process_batch(lines, img_path, inputH, inputW, train=False)
    y_val = to_categorical(np.array(y_labels), num_classes)
    return x_val, y_val


# Output directory for saving results
outputdir = 'result/'
if os.path.isdir(outputdir):
    print('save in :' + outputdir)
else:
    os.makedirs(outputdir)

# Define paths and parameters
train_img_path = './MIT-BIH_AD/'
test_img_path = './MIT-BIH_AD/'
train_file = 'MIT-BIH_AD_train.txt'
test_file = 'MIT-BIH_AD_val.txt'
num_classes = 5

# Read the training and validation samples
f1 = open(train_file, 'r')
f2 = open(test_file, 'r')
lines=f1.readlines()
f1.close()
train_samples=len(lines)
lines=f2.readlines()
f2.close()
val_samples=len(lines)

batch_size = 32
epochs = 100
input_h = 128
input_w = 128

# Load model
model = proposed_model(nb_classes=num_classes)

# Configure Adam optimizer
lr = 0.0001
adam = Adam(learning_rate=lr)  # Correct usage
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Callbacks for TensorBoard, ModelCheckpoint, and EarlyStopping
callbacks = [
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
    ModelCheckpoint('./result/mit_bih_2D.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)  # Early stopping callback 10轮无提升则停止
]

# # Train the model
# history = model.fit(
#     generator_train_batch(train_file, batch_size, num_classes, train_img_path, input_h, input_w),
#     steps_per_epoch=train_samples // batch_size,
#     epochs=epochs,
#     callbacks=callbacks,
#     validation_data=generator_val_batch(test_file, batch_size, num_classes, test_img_path, input_h, input_w),
#     validation_steps=val_samples // batch_size,
#     verbose=1
# 替换为

# Load full validation set
x_val, y_val = load_val_data(test_file, test_img_path, input_h, input_w, num_classes)

# Train the model
history = model.fit(
    generator_train_batch(train_file, batch_size, num_classes, train_img_path, input_h, input_w),
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_val, y_val),  # 使用固定验证集
    verbose=1
)


# Plot and save history
plot_history(history, outputdir)
save_history(history, outputdir)