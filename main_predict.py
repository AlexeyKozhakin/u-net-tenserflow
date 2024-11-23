import os
# Инструменты для работы с изображениями
from tensorflow.keras.preprocessing import image
import numpy as np
from libs import (PRED_DIRECTORY,
                  MODEL_DIR, MODEL_FILE,
                  process_images_predict_save)
from opt_data_loader.libs_opt_data_loader import load_images_for_prediction, BATCH_SIZE

import tensorflow as tf
import json

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



print(PRED_DIRECTORY)


pred_dataset = load_images_for_prediction(
    os.path.join(PRED_DIRECTORY, 'original'))
# Создание батчей
pred_dataset = pred_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


model_unet = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL_FILE))
# Пример использования
filenames = sorted(os.listdir(f'{PRED_DIRECTORY}/original'))  # Загружаем имена файлов
# Запуск предсказаний и сохранение результатов
process_images_predict_save(
    model=model_unet,
    x_pred=pred_dataset,
    filenames=filenames,
    save_dir=os.path.join(PRED_DIRECTORY, 'segment')
)
