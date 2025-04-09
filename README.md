# Eng
# 🍓 Emotion Recognition with CNN - Machine Learning Project

**Emotion Recognition** with CNN is a machine learning project focused on classifying emotions from facial images using a Convolutional Neural Network (CNN). The goal is to predict emotions such as anger, happiness, sadness, and more, based on images from the FER-2013 dataset. This project can be applied in areas such as sentiment analysis, customer feedback, and user experience improvement.

##📝 Description

This project develops a deep learning model that classifies facial expressions into 7 distinct emotions:

- Anger

- Disgust

- Fear

- Happiness

- Sadness

- Surprise

- Neutral

The model is trained using the FER-2013 dataset and a CNN architecture. It aims to provide accurate emotion recognition for various applications, such as human-computer interaction, psychological studies, and emotion-driven marketing.

## 📁 Project Structure

- `train/` — Folder with training images organized by emotion categories.

- `test/` — Folder with test images.

- `emotion.h5` — Saved model file for predictions and further use.

- `model.py` — Python script containing data preprocessing, model creation, and training steps.

- `README.md` — Project documentation.

## 🛠 Technologies Used

- Python — Main programming language

- TensorFlow & Keras — Deep learning framework

- OpenCV — Image processing library

- NumPy & Pandas — Data manipulation libraries

- Matplotlib & Seaborn — Data visualization libraries

- Scikit-learn — For label encoding and model evaluation

## 💻 How to Use

- Install Dependencies:Install the required libraries using the following command: pip install numpy pandas matplotlib seaborn opencv-python tensorflow

- Train the Model: Run the model.py script to train the emotion recognition model on the FER-2013 dataset.

- Save and Load Model: Once training is complete, the model is saved as emotion.h5 for later use.

- Visualize Results: Training loss and accuracy can be visualized using Matplotlib.

## 🔗 Download the Dataset

You can download the FER-2013 dataset from Kaggle using the following link:

[FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)


# Rus
## 🍓 Распознавание эмоций с помощью CNN — Проект машинного обучения
Распознавание эмоций с помощью CNN — это проект машинного обучения, ориентированный на классификацию эмоций по изображениям лиц с использованием сверточной нейронной сети (CNN). Цель проекта — предсказать эмоции, такие как гнев, счастье, печаль и другие, на основе изображений из датасета FER-2013. Этот проект может быть применен в таких областях, как анализ настроений, обратная связь от пользователей и улучшение пользовательского опыта.

## 📝 Описание
Этот проект разрабатывает модель глубокого обучения для классификации выражений лиц на 7 различных эмоций:

- Гнев

- Отвращение

- Страх

- Счастье

- Печаль

- Удивление

- Нейтральное состояние

Модель обучена на датасете FER-2013 и использует архитектуру CNN. Она направлена на точное распознавание эмоций, что может быть полезно в таких областях, как взаимодействие человек-компьютер, психологические исследования и маркетинг, ориентированный на эмоции.

## 📁 Структура проекта
- `train/` — Папка с обучающими изображениями, организованными по категориям эмоций.

- `test/` — Папка с тестовыми изображениями.

- `emotion.h5` — Сохраненная модель для предсказаний и дальнейшего использования.

- `model.py` — Скрипт на Python, содержащий предобработку данных, создание модели и этапы обучения.

- `README.md` — Документация проекта.

## 🛠 Используемые технологии
- Python — Основной язык программирования

- TensorFlow & Keras — Фреймворк для глубокого обучения

- OpenCV — Библиотека для обработки изображений

- NumPy & Pandas — Библиотеки для обработки данных

- Matplotlib & Seaborn — Библиотеки для визуализации данных

- Scikit-learn — Для кодирования меток и оценки модели

## 💻 Как использовать
- Установить зависимости: Установите необходимые библиотеки с помощью следующей команды: pip install numpy pandas matplotlib seaborn opencv-python tensorflow

- Обучить модель: Запустите скрипт model.py, чтобы обучить модель распознавания эмоций на датасете FER-2013.

- Сохранить и загрузить модель: После завершения обучения модель сохраняется в файл emotion.h5 для дальнейшего использования.

- Визуализировать результаты: Потери и точность обучения можно визуализировать с помощью Matplotlib.

## 🔗 Скачать датасет
Вы можете скачать датасет FER-2013 с Kaggle по следующей ссылке:

[FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)
