import os
# Инструменты для работы с изображениями
from tensorflow.keras.preprocessing import image
import numpy as np
from libs import (PRED_DIRECTORY, load_imageset,
                  MODEL_DIR, MODEL_FILE,
                  process_images_predict_save)

import tensorflow as tf
import json

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



print(PRED_DIRECTORY)

pred_images = load_imageset(PRED_DIRECTORY, 'original', 'Обучающая')
# Формирование обучающей выборки
x_pred = []                          # Cписок под обучающую выборку
for img in pred_images:              # Для всех изображений выборки:
    x = image.img_to_array(img)       # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_pred.append(x)                 # Добавление элемента в x_train
x_pred = np.array(x_pred)           # Перевод всей выборки в numpy
print(x_pred.shape)                  # Форма x_train


model_unet = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_FILE))
# Пример использования
filenames = sorted(os.listdir(f'{PRED_DIRECTORY}/original'))  # Загружаем имена файлов
process_images_predict_save(model_unet, x_pred, filenames, save_dir= os.path.join(PRED_DIRECTORY,'segment'))
