# Проект мультиклассовой классификации текстов

Этот проект реализует систему мультиклассовой классификации текстов с использованием нескольких моделей (SimpleNN, RandomForest, Logistic Regression и предварительно обученного Трансформера). Он включает бэкенд на FastAPI для предоставления предсказаний и фронтенд на Streamlit для взаимодействия с пользователем и отображения метрик оценки моделей.

<img src="https://github.com/user-attachments/assets/a8919c8e-52f1-47e2-8b52-338905bb84c8" width="800" alt="Описание изображения">

## Возможности
Несколько моделей:
 - Простая нейронная сеть (TensorFlow/Keras)
 - Случайный лес (scikit-learn)
 - Логистическая регрессия (scikit-learn)
 - Предварительно обученный Трансформер (Hugging Face Transformers)

API: Бэкенд на FastAPI для обслуживания предсказаний моделей.

Фронтенд: Приложение на Streamlit для ввода текста, отображения предсказаний от всех моделей, оценки моделей и просмотра метрик.

Обучение и Оценка: Скрипты для обучения моделей и оценки всех моделей.

Модульность: Код организован в директории src, api, frontend. Код покрыт тестами

## Структура проекта
.
├── api/                    # Приложение FastAPI
│   ├── main.py             # Основная логика API
│   ├── schemas.py          # Схемы Pydantic
│   └── utils.py            # Вспомогательные функции API (например, для рейтингов)
├── data/                   # Каталог для обучающих данных (CSV файлы)
├── frontend/               # Приложение Streamlit (фронтенд)
│   └── app.py
├── models/                 # Сохраненные файлы моделей
│   ├── simple_nn_model/
│   ├── random_forest_model/
│   ├── logistic_regression_model/
│   └── llm_models/         # Каталог для предварительно обученной трансформерной модели
├── reports/                # Отчеты об оценке
│   └── evaluation_metrics.json
├── src/                    # Основной исходный код
│   ├── config/
│   │   └── config.py       # Конфигурация (пути, метки и т.д.)
│   ├── data/
│   │   └── dataset.py      # Загрузка и предварительная обработка данных
│   ├── models/             # Определения моделей
│   │   ├── base_model.py
│   │   ├── simple_nn.py
│   │   ├── random_forest_model.py
│   │   ├── logistic_regression_model.py
│   │   └── transformer_model.py
│   └── utils/
│       └── metrics.py      # Расчет метрик
├── static/                 # Статические файлы для веб-интерфейса FastAPI
│   └── style.css
├── templates/              # HTML шаблоны для веб-интерфейса FastAPI
│   └── index.html
├── tests/                  # Тесты Pytest
│   ├── test_api.py
│   └── test_models.py
├── .gitignore
├── evaluate.py             # Скрипт для оценки моделей
├── train.py                # Скрипт для обучения моделей
└── README.md
