from PIL import Image
from tensorflow.keras.preprocessing import image
# Инструменты для работы с массивами
import numpy as np
# Системные инструменты
import time
import os
import json


current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Чтение конфигурации из JSON
with open("config.json", "r") as f:
    config = json.load(f)

# Использование переменных
DATA_DIR = config["DATA_DIR"]
MODEL_DIR = config["MODEL_DIR"]
MODEL_FILE = config["MODEL_FILE"]


# 20 - 1 (16) - 1 (17) - 1 (8) = 17 классов
class_colors_stpls3d = {
    0: (0, 0, 0),  # Ground - Черный
    1: (128, 128, 128),  # Building - Зеленый           1==17
    2: (255, 255, 0),  # LowVegetation - Желтый
    3: (0, 0, 255),  # MediumVegetation - Синий
    4: (255, 0, 0),  # HighVegetation - Красный
    5: (0, 255, 255),  # Vehicle - Бирюзовый
    6: (255, 0, 255),  # Truck - Магента
    7: (255, 128, 0),  # Aircraft - Оранжевый
    9: (255, 20, 147),  # Bike - Deep Pink
    10: (255, 69, 0),  # Motorcycle - Красный Апельсин
    11: (210, 180, 140),  # LightPole - Бежевый
    12: (255, 105, 180),  # StreetSign - Hot Pink
    13: (165, 42, 42),  # Clutter - Коричневый
    14: (139, 69, 19),  # Fence - Темно-коричневый
    15: (128, 0, 128),  # Road - Фиолетовый
    18: (222, 184, 135),  # Dirt - Седой
    19: (127, 255, 0),  # Grass - Ярко-зеленый
}

CLASS_LABELS = list(class_colors_stpls3d.values())

CLASS_COUNT = len(CLASS_LABELS)


IMG_WIDTH = 512               # Ширина картинки
IMG_HEIGHT = 512              # Высота картинки
N_CHANNELS = 3

TRAIN_DIRECTORY = os.path.join(DATA_DIR,'train')     # Название папки с файлами обучающей выборки
VAL_DIRECTORY = os.path.join(DATA_DIR,'val')         # Название папки с файлами проверочной выборки
TEST_DIRECTORY = os.path.join(DATA_DIR,'test')
PRED_DIRECTORY = os.path.join(DATA_DIR,'predict')




def load_imageset(folder,   # имя папки
                  subset,   # подмножество изображений - оригинальные или сегментированные
                  title     # имя выборки
                  ):

    # Cписок для хранения изображений выборки
    image_list = []

    # Отметка текущего времени
    cur_time = time.time()

    # Для всех файлов в каталоге по указанному пути:
    for filename in sorted(os.listdir(f'{folder}/{subset}')):

        # Чтение очередной картинки и добавление ее в список изображений требуемого размера
        image_list.append(image.load_img(os.path.join(f'{folder}/{subset}', filename),
                                         target_size=(IMG_WIDTH, IMG_HEIGHT)))

    # Вывод времени загрузки картинок выборки
    print('{} выборка загружена. Время загрузки: {:.2f} с'.format(title,
                                                                  time.time() - cur_time))

    # Вывод количества элементов в выборке
    print('Количество изображений:', len(image_list))

    return image_list


# Функция преобразования цветного сегментированного изображения в метки классов

def rgb_to_labels(image_list  # список цветных изображений
                 ):

    result = []

    # Для всех картинок в списке:
    for d in image_list:
        sample = np.array(d)
        # Создание пустой 1-канальной картики
        y = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype='uint8')

        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение 3-х канальных пикселей классов и занесение метки класса
            y[np.where(np.all(sample == CLASS_LABELS[i], axis=-1))] = i

        result.append(y)

    return np.array(result)

# Функция преобразования тензора меток класса в цветное сегметрированное изображение

import tensorflow as tf

def rgb_to_labels_tf(image_list):
    """
    Преобразует список цветных изображений в метки классов для TensorFlow.
    """
    result = []

    for d in image_list:
        # Преобразование в uint8, если входной тип float32
        if d.dtype != tf.uint8:
            d = tf.cast(d, tf.uint8)

        sample = tf.convert_to_tensor(d, dtype=tf.uint8)

        # Создание пустой 1-канальной картинки
        y = tf.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.uint8)

        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение 3-х канальных пикселей классов и установка метки
            mask = tf.reduce_all(sample == cl, axis=-1)
            y = tf.where(mask[..., None], i, y)

        result.append(y)

    return tf.stack(result)




def labels_to_rgb(image_list  # список одноканальных изображений
                 ):

    result = []

    # Для всех картинок в списке:
    for y in image_list:
        # Создание пустой цветной картики
        temp = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype='uint8')

        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение пикселов класса и заполнение цветом из CLASS_LABELS[i]
            temp[np.where(np.all(y==i, axis=-1))] = CLASS_LABELS[i]

        result.append(temp)

    return np.array(result)

# Функция визуализации процесса сегментации изображений и сохранения результатов
def process_images_predict_save(
        model,  # обученная модель
        x_pred,  # tf.data.Dataset с изображениями для предсказания
        filenames,  # список файлов, по которым загружены изображения
        save_dir='predictions'  # директория для сохранения изображений
):
    """
    Процесс предсказания, визуализации и сохранения результатов.
    """

    import numpy as np
    from PIL import Image

    # Создание директории, если её не существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Генерация предсказаний с помощью модели
    predictions = model.predict(x_pred)  # Предсказываем все данные из tf.data.Dataset

    # Индексирование классов
    predicted_classes = np.argmax(predictions, axis=-1)  # Конвертируем one-hot в индексы классов

    # Преобразование предсказаний в RGB (используйте вашу функцию labels_to_rgb)
    predicted_images = labels_to_rgb(predicted_classes[..., None])  # Конвертация в RGB

    # Сохранение предсказанных изображений с использованием имён файлов
    for i, filename in enumerate(filenames):
        # Преобразуем numpy-данные в изображение
        pred_image = Image.fromarray(predicted_images[i].astype(np.uint8))

        # Извлечение имени файла и сохранение результата
        base_filename = os.path.basename(filename)  # Извлекаем имя файла
        pred_image.save(os.path.join(save_dir, base_filename))  # Сохраняем изображение

    print(f"Предсказанные изображения сохранены в директории: {save_dir}")
