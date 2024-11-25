import tensorflow as tf
import numpy as np
from libs import class_colors_stpls3d
from opt_data_loader.libs_opt_data_loader import parse_mask
# Настройки для теста
IMG_HEIGHT = 256  # Задайте желаемую высоту изображения
IMG_WIDTH = 256   # Задайте желаемую ширину изображения
CLASS_LABELS = tf.constant(
list(class_colors_stpls3d.values())
, dtype=tf.uint8)


# Укажите путь к тестовой папке с масками
mask_folder = r"D:\data\data_for_training\data_training_stpl3d_256_512\train\segment"  # Замените на путь к папке с масками
mask_files = tf.io.gfile.glob(f"{mask_folder}/*.png")  # Загрузить все PNG-файлы в папке

print(f"Найдено {len(mask_files)} масок")

for mask_path in mask_files:
    print(f"\nОбрабатывается маска: {mask_path}")

    # Преобразуем маску
    mask = parse_mask(mask_path)

    # Извлекаем уникальные классы
    unique_classes = np.unique(mask.numpy())

    # Выводим результат
    print(f"Найденные уникальные классы в маске {mask_path}: {unique_classes}")
