# локально
#Должен быть python 3.9 на железе

py -3.9 -m venv .venv

.venv\Scripts\Activate

pip install -r requirements.txt

#обучение моделей

python train.py

#запус локального сервера

uvicorn api.main:app --reload --port 8001



# или через Docker
docker-compose up --build
