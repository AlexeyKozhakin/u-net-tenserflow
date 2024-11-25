import os
import tensorflow as tf
from libs import class_colors_stpls3d
import json


# Чтение конфигурации из JSON
with open('config.json', "r") as f:
    config = json.load(f)

BATCH_SIZE = int(config["BATCH_SIZE"])
N_CHANNELS = int(config["N_CHANNELS"])
IMG_HEIGHT = int(config["IMG_HEIGHT"])
IMG_WIDTH = int(config["IMG_WIDTH"])

CLASS_LABELS = tf.constant(
list(class_colors_stpls3d.values())
, dtype=tf.uint8)
# print(CLASS_LABELS)

# Функция для загрузки данных с использованием tf.data.Dataset
def parse_image(image_path):
    """
    Загрузка и предобработка изображений.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=N_CHANNELS)  # Для цветных изображений
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0  # Нормализация в диапазон [0, 1]
    return image


# def parse_mask(mask_path):
#     """
#     Загрузка и предобработка масок.
#     """
#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels=3)  # Для цветных масок
#     mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])  # Масштабируем до требуемого размера
#     mask = tf.cast(mask, tf.uint8)
#
#     # Создаем маску, сравнивая с CLASS_LABELS
#     mask = tf.equal(mask[..., None, :], CLASS_LABELS)  # Создаем one-hot вектор
#     mask = tf.reduce_any(mask, axis=-1)  # Приводим к бинарным значениям (по каждому классу)
#
#     # Преобразуем one-hot маску в индексы классов
#     mask = tf.argmax(mask, axis=-1)  # Приводим к формату (height, width)
#
#     print(f"Mask shape after argmax: {mask.shape}")  # Должно быть (height, width)
#     return mask

def parse_mask(mask_path):
    """
    Загрузка и предобработка цветных масок.
    """
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)  # Загружаем маску как цветное изображение
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])  # Масштабируем до требуемого размера
    mask = tf.cast(mask, tf.uint8)  # Приводим к целочисленному типу

    # Создаем one-hot encoding на основе сравнения с цветовыми метками
    mask = tf.equal(mask[..., None, :], CLASS_LABELS)  # Сравниваем пиксель со списком классов
    mask = tf.reduce_all(mask, axis=-1)  # Убираем цветовые каналы, оставляя бинарное представление
    mask = tf.cast(mask, tf.float32)  # Переводим в float32, если нужно для обучения

    print(f"Mask shape after one-hot encoding: {mask.shape}")  # Должно быть (height, width, num_classes)
    return mask




def load_dataset(image_dir, mask_dir):
    """
    Создание tf.data.Dataset из директорий.
    """
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, mask: (parse_image(img), parse_mask(mask)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def load_images_for_prediction(image_dir):
    """
    Загрузка и предобработка изображений для предсказания.
    """
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset



