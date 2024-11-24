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



# Регистрируем метрику
@tf.keras.utils.register_keras_serializable()
def class_accuracy(class_id):
    def accuracy(y_true, y_pred):
        y_true_class = tf.cast(tf.equal(y_true, class_id), tf.float32)  # Метки для конкретного класса
        y_pred_class = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), class_id), tf.float32)  # Прогнозы для класса
        correct_predictions = tf.reduce_sum(y_true_class * y_pred_class)  # Корректные предсказания
        total_true = tf.reduce_sum(y_true_class)  # Общее количество истинных меток для класса
        return correct_predictions / (total_true + tf.keras.backend.epsilon())  # Избегаем деления на 0
    return accuracy


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
