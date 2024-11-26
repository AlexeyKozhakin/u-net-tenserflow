import tensorflow as tf
import numpy as np
from libs import class_colors_stpls3d
from opt_data_loader.libs_opt_data_loader import parse_image

# Укажите путь к тестовой папке с масками
img_folder = r"D:\data\data_for_training\data_training_stpl3d_64_512\train\original"  # Замените на путь к папке с масками
img_files = tf.io.gfile.glob(f"{img_folder}/*.png")  # Загрузить все PNG-файлы в папке

print(f"Найдено {len(img_files)} масок")

for img_path in img_files[:1]:
    print(f"\nОбрабатывается маска: {img_path}")

    # читаем картинку
    image = parse_image(img_path)

    # Выводим результат
print(image.shape)
print(image[:,0,1])
