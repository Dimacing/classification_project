# Проект мультиклассовой классификации текстов
# Реализует систему классификации текстов с моделями (SimpleNN, RandomForest, Logistic Regression, Трансформер),
# FastAPI бэкендом и Streamlit фронтендом.

# --- Локальная настройка и запуск ---

# 1. Предварительные требования
#    - Python (рекомендуется 3.8+)
#    - pip и venv

# 2. Клонирование репозитория
git clone <URL-репозитория>
cd <имя-репозитория>

# 3. Создание и активация виртуального окружения
#    (Пример для Windows с Python 3.9, если он в PATH как 'py -3.9')
#    (Для Linux/macOS: python3.9 -m venv .venv && source .venv/bin/activate)
py -3.9 -m venv .venv
.venv\Scripts\Activate

# 4. Установка зависимостей
#    (Убедитесь, что файл requirements.txt существует и содержит все пакеты)
pip install -r requirements.txt

# 5. Подготовка данных
#    - Создайте директорию `data/` в корне проекта.
#    - Поместите ваши CSV-файлы с обучающими данными (1.csv, 2.csv, ..., 6.csv) в `data/`.
#      Каждый CSV должен содержать столбцы `doc_text`, `image2text`, `speech2text`.
#    - (Опционально) Для модели Трансформер:
#      - Создайте `models/llm_models/`.
#      - Поместите файлы вашей pre-trained Hugging Face модели
#        (включая `pytorch_model.bin`, `config.json`, файлы токенизатора
#        и `label_binarizer.pkl`) в `models/llm_models/`.

# 6. Обучение моделей (SimpleNN, RandomForest, Logistic Regression)
python train.py
#    Обученные модели сохраняются в директорию `models/`.

# 7. Оценка моделей
python evaluate.py
#    Метрики сохраняются в `reports/evaluation_metrics.json`.
#    Примечание: для 'transformer' метрики могут быть фейковыми по умолчанию.

# 8. Запуск API сервера (FastAPI)
#    (Порт 8003 используется по умолчанию в frontend/app.py, можно изменить)
uvicorn api.main:app --reload --port 8003
#    API будет доступен по адресу: http://localhost:8003
#    Документация API (Swagger): http://localhost:8003/docs

# 9. Запуск фронтенда (Streamlit)
#    (Убедитесь, что API сервер запущен)
streamlit run frontend/app.py
#    Фронтенд обычно доступен по адресу: http://localhost:8501

# 10. Запуск тестов (опционально)
#     (Требуется pytest)
pytest

# --- Запуск через Docker ---
# (Требуется наличие Dockerfile и docker-compose.yml в проекте)
docker-compose up --build
