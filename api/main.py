import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.models.simple_nn import SimpleNNModel
from src.models.distilbert_model import DistilBERTModel
from src.config.config import MODEL_DIR, LABELS
from api.schemas import TextInput, ModelRatingInput
from api.utils import add_rating, get_average_ratings

app = FastAPI(
    title="Text Multi-label Classifier API",
    description="Классификация текста по темам: спорт, юмор, реклама, соцсети, политика, личная жизнь",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# !!!!Загружаем модели
simple_nn_model = SimpleNNModel()
simple_nn_model.load(MODEL_DIR / "simple_nn_model")

distilbert_model = DistilBERTModel(num_classes=len(LABELS))
distilbert_model.load(MODEL_DIR / "distilbert_model")

# !!!!Прописываем модели
models = {
    "simple_nn": simple_nn_model,
    "distilbert": distilbert_model
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    text: str = Form(None),
    file: UploadFile = File(None)
):
    if file and file.filename:
        content = await file.read()
        text = content.decode("utf-8")

    if not text:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Введите текст или загрузите файл"
        })

    results = {}
    for model_name, model in models.items():
        preds = model.predict([text])[0]
        preds_dict = {label: float(pred) for label, pred in zip(LABELS, preds)}
        results[model_name] = preds_dict

    ratings = get_average_ratings()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "ratings": ratings
    })


@app.post("/rate_model_web")
async def rate_model_web(
    model_name: str = Form(...),
    rating: int = Form(...)
):
    add_rating(model_name, rating)
    return RedirectResponse("/", status_code=303)


@app.post("/predict_text")
async def predict_text(input_data: TextInput):
    results = {}

    for model_name, model in models.items():
        preds = model.predict([input_data.text])[0]
        preds_dict = {label: float(pred) for label, pred in zip(LABELS, preds)}
        results[model_name] = preds_dict

    return results


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    return await predict_text(TextInput(text=text))


@app.post("/rate_model")
async def rate_model(input_data: ModelRatingInput):
    add_rating(input_data.model_name, input_data.rating)
    return {"message": f"Рейтинг {input_data.rating} сохранен для {input_data.model_name}"}


@app.get("/model_ratings")
async def model_ratings():
    return get_average_ratings()


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
