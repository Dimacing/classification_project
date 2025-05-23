# Проект мультиклассовой классификации текстов

Этот проект реализует систему мультиклассовой классификации текстов с использованием нескольких моделей (SimpleNN, RandomForest, Logistic Regression и предварительно обученного Трансформера). Он включает бэкенд на FastAPI для предоставления предсказаний и фронтенд на Streamlit для взаимодействия с пользователем и отображения метрик оценки моделей.

<img src="https://github.com/user-attachments/assets/a8919c8e-52f1-47e2-8b52-338905bb84c8" width="800" alt="Описание изображения">

## Видео-демонстрация разработанного решения

https://github.com/user-attachments/assets/2d95c6bd-b9b2-433d-b725-be544ca2f6bd



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

## EDA
Открыть файл EDA_class_text.html как страницу

<img src="https://github.com/user-attachments/assets/6e61f04d-26a1-4514-8bc4-5d1374d4cc0f" width="600" alt="Описание изображения">

##  Настройка и Установка
1. Предварительные требования
- Python (рекомендуется 3.9)
- pip и venv
2. Создайте и активируйте виртуальное окружение
```text
# Для Windows
py -3.9 -m venv .venv

.venv\Scripts\Activate
```
3. Установите зависимости
```text
pip install -r requirements.txt
```
4. Обучение моделей (не обязятельно, в репеозитории уже лежат обученные модели на маленьком датасете)
```text
python train.py
```
5. Оценка моделей (не обязательно, есть уже посчитанные метрики)
```text
python evaluate.py
```

6. Запуск API сервера 
```text
uvicorn api.main:app --host 0.0.0.0 --port 8003 --reload
```

7. Запуск фронтенда Streamlit (параллельно в другом окне командной строки)
```text
streamlit run frontend/app.py
```
- Фронтенд обычно будет доступен по адресу http://localhost:8501.
- Фронтенд взаимодействует с API сервером (убедитесь, что API запущен).

8. Запуск тестов (Не обязательно, некоторые тесты требуют доработки)
```text
pytest
```


## Структура проекта

```text
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


