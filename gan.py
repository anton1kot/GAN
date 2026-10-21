import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.layers import LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_data():
    # загружаем данные
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # нормируем данные
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    # превращаем изображения в плоские вектора размером в 784
    # 60000 — это количество изображений
    X_train = X_train.reshape(60000, 784)

    return X_train

def create_generator():
    generator=Sequential()

    # первый скрытый слой из 256 нейронов получает 100 чисел
    generator.add(Dense(units=256,input_dim=100))

    # функция активации LeakyReLU первого слоя 
    # умножает отрицательные значения на 0.2
    generator.add(LeakyReLU(0.2))

    # второй скрытый слой из 512 нейронов
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))

    # третий скрытый слой из 1024 нейронов
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))

    # выходной слой из 784 нейронов (размер картинки) 
    # с функцией активации Tahn
    generator.add(Dense(units=784, activation='tanh'))

    # стандартные настройки функции потери и оптимизатора
    generator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    return generator

def create_discriminator():
    discriminator=Sequential()

    # первый слой из 1024 нейронов получает 784 чисел
    discriminator.add(Dense(units=1024, input_dim=784))

    # добавляем функцию активации LeakyReLU
    discriminator.add(LeakyReLU(0.2)) 

    # добавляем дополнительный слой
    # он обнуляет случайные нейроны, чтобы избежать переобучения
    discriminator.add(Dropout(0.3)) 

    # второй слой из 512 нейронов
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    # третий слой из 256 нейронов
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))

    # выходной слой из одного нейрона с функцией активации — сигмоидой
    discriminator.add(Dense(units=1, activation='sigmoid'))

    # стандартные настройки обучения
    discriminator.compile(loss='binary_crossentropy', 
                          optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    return discriminator

def create_gan(discriminator, generator):
    # замораживаем обучение дискриминатора
    discriminator.trainable = False
    # создаем входные данные
    gan_input = Input(shape=(100,))
    # запускаем генератор
    x = generator(gan_input)
    # проходимся дискриминатором по сгенерированным данным 
    gan_output = discriminator(x)
    # создаем модель
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', 
                optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    return gan

def plot_generated_images(gan, generator, epoch=None):
    # создаем входные данные для генератора
    # 16 — это количество картинок
    noise = np.random.normal(0, 1, size=(16, gan.input_shape[1]))
    
    # генерируем изображения
    generated_images = generator.predict(noise)
    
    # нормируем данные так, чтобы они были от 0 до 1
    generated_images = (generated_images + 1) / 2.0
    
    # создаем фигуру
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    # создаем все картинки
    for i, ax in enumerate(axes.flat):
        # превращаем плоский вектор изображения в матрицу 28 на 28
        img = generated_images[i].reshape(28, 28)

        # добавляем ч/б картинку
        ax.imshow(img, cmap='gray')

        # убираем оси графиков
        ax.axis('off')
    
    # сохраняем в файл
    if epoch is not None:
        filename = f'generated_epoch_{epoch}.png'
        plt.savefig(filename)
        print(f'Сохранено: {filename}')
    
    plt.show()

def train(epochs, batch_size):
    # загружаем входные данные
    X_train = load_data()

    # количество кусков данных
    batch_count = X_train.shape[0] // batch_size

    # создаем модель
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    # в процессе обучения происходит несколько циклов (эпох)
    # в одном цикле мы один раз обучаем дискриминатор, фиксируем его веса
    # и обучаем генератор
    for e in range(1, epochs + 1):

        # бьем данные для обучения на куски и обучаемся на каждом отдельно
        for _ in range(batch_count):
            # создаем случайный шум на вход генератора
            noise = np.random.normal(0, 1, [batch_size, 100])

            # генерируем фейковые изображения с помощью генератора
            generated_images = generator.predict(noise)

            # берем случайные настоящие изображения из данных для обучения
            image_batch = X_train[np.random.randint(low=0, 
                                                    high=X_train.shape[0], 
                                                    size=batch_size)]

            # объединяем настоящие и сгенерированные изображения
            X= np.concatenate([image_batch, generated_images])

            # создаем подписи к изображениям 
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            # обучаем дискриминатор на получившейся выборке
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)

            # снова создаем случайный шум для подачи на вход генератору
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # замораживаем веса дискриминатора
            discriminator.trainable=False

            # обучаем модель
            gan.train_on_batch(noise, y_gen)

        # выводим картинку каждые 5 эпох
        if e % 5 == 0:
            print(f'Эпоха {e}/{epochs}')
            plot_generated_images(gan, generator, epoch=e)
            input('Закройте окно с картинкой и нажмите Enter, чтобы продолжить...')


# ---------- ввод количества эпох ----------
print('Введите количество эпох для обучения:')
print('  10 эпох  — ~2-3 минуты (цифры ещё размытые)')
print('  20 эпох  — ~4-6 минут (цифры узнаваемы)')
print('  50 эпох  — ~10-15 минут (цифры чёткие)')
print('  100 эпох — ~20-30 минут (хорошее качество)')
epochs_input = input('Ваш выбор: ')
try:
    epochs = int(epochs_input)
    if epochs < 1:
        raise ValueError
except ValueError:
    print('Некорректное число, используется 20 эпох')
    epochs = 20

train(epochs, 128)
