import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from libs import (
    TRAIN_DIRECTORY, VAL_DIRECTORY,
    CLASS_COUNT
)
from opt_data_loader.libs_opt_data_loader import load_dataset, IMG_WIDTH, IMG_HEIGHT
from model import unet
import json
import tensorflow as tf
from model import class_accuracy

# # Получаем абсолютный путь к текущему файлу (запускаемому скрипту)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
#
# # Формируем путь к файлу config.json
# config_path = os.path.join(current_dir, '..', 'config.json')

# Чтение конфигурации из JSON
with open('config.json', "r") as f:
    config = json.load(f)

BATCH_SIZE = int(config["BATCH_SIZE"])
N_CHANNELS = int(config["N_CHANNELS"])
MODEL_DIR = config["MODEL_DIR"]
PRE_TRIANED_MODEL_FILE = config["PRE_TRIANED_MODEL_FILE"]
# Настройка данных

train_dataset = load_dataset(
    os.path.join(TRAIN_DIRECTORY, 'original'),
    os.path.join(TRAIN_DIRECTORY, 'segment')
)
val_dataset = load_dataset(
    os.path.join(VAL_DIRECTORY, 'original'),
    os.path.join(VAL_DIRECTORY, 'segment')
)

# Добавление батчей, перемешивания и оптимизаций
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

if PRE_TRIANED_MODEL_FILE:
    from tensorflow.keras.models import load_model
    # Путь к сохранённой модели
    model_load_path = os.path.join(MODEL_DIR, PRE_TRIANED_MODEL_FILE)  # Укажите путь к вашей модели

    # Загрузка модели
    model_unet = load_model(model_load_path)

    print("Модель успешно загружена!")
else:
    # Создание модели
    model_unet = unet(CLASS_COUNT, (IMG_WIDTH, IMG_HEIGHT, N_CHANNELS))  # 4-канальное изображение

# Определение путей для сохранения модели
model_save_path = os.path.join(MODEL_DIR, 'model_exp_optimized.{epoch:02d}.keras')

# Настройка колбеков
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
mcp_save = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')


# class_weights = {1: 2.0, 2:4.0, 3:10.0, 4:4.0, 5:10.0, 6:10.0, 7:10.0, 8:10.0, 9:10.0, 10:10.0,
#                  11:10.0, 12:10.0, 13:10.0, 14:10.0, 15:3.0, 16:1.0, 17:3.0}  # Пример весов для классов
# Обучение модели
history = model_unet.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stopping, mcp_save]
)