import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np
import json
from src.models.simple_nn import SimpleNNModel
from src.models.random_forest_model import RandomForestModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.transformer_model import TransformerModel
from src.config.config import MODEL_DIR, LABELS
from api.schemas import TextInput, ModelRatingInput
from api.utils import add_rating, get_average_ratings

app = FastAPI(
    title="Text Multi-label Classifier API",
    description=f"Классификация текста по темам: {', '.join(LABELS)}",
    version="1.5"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


EVALUATION_METRICS_PATH = Path("./reports/evaluation_metrics.json")

# -
MODEL_DIR = Path(MODEL_DIR)
LLM_MODEL_SAVE_PATH = MODEL_DIR / "llm_models"
app.state.models = {}

try:
    simple_nn_model_path = MODEL_DIR / "simple_nn_model"; simple_nn_model = SimpleNNModel(); simple_nn_model.load(simple_nn_model_path)
    app.state.models["simple_nn"] = simple_nn_model; print("SimpleNN model loaded.")
except Exception as e: print(f"Error loading SimpleNN: {e}")

try:
    rf_model_dir = MODEL_DIR / "random_forest_model"; rf_model_path = rf_model_dir / "model.joblib"; random_forest_model = RandomForestModel(num_classes=len(LABELS)); random_forest_model.load(rf_model_path)
    app.state.models["random_forest"] = random_forest_model; print("RandomForest model loaded.")
except Exception as e: print(f"Error loading RandomForest: {e}")

try:
    logreg_model_dir = MODEL_DIR / "logistic_regression_model"; logreg_model_path = logreg_model_dir / "model.joblib"; logreg_model = LogisticRegressionModel(num_classes=len(LABELS)); logreg_model.load(logreg_model_path)
    app.state.models["logistic_regression"] = logreg_model; print("Logistic Regression model loaded.")
except Exception as e: print(f"Error loading Logistic Regression: {e}")

try:
    if not LLM_MODEL_SAVE_PATH.is_dir(): raise FileNotFoundError(f"Transformer dir not found: {LLM_MODEL_SAVE_PATH}")
    transformer_model = TransformerModel(model_path=LLM_MODEL_SAVE_PATH); transformer_model.load()
    app.state.models["transformer"] = transformer_model; print("Transformer model loaded.")
except Exception as e: print(f"Error loading Transformer: {e}")

print(f"Models loaded: {list(app.state.models.keys())}")



@app.get("/evaluation_metrics")
async def get_evaluation_metrics():
    if not EVALUATION_METRICS_PATH.is_file():
        print(f"Warning: Evaluation metrics file not found at {EVALUATION_METRICS_PATH}")
        raise HTTPException(status_code=404, detail="Evaluation metrics not found. Run evaluate.py first.")
    try:
        with open(EVALUATION_METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        return JSONResponse(content=metrics_data)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {EVALUATION_METRICS_PATH}")
        raise HTTPException(status_code=500, detail="Error reading evaluation metrics file.")
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        raise HTTPException(status_code=500, detail="Could not load evaluation metrics.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    available_models = list(request.app.state.models.keys())
    current_ratings = get_average_ratings()
    return templates.TemplateResponse("index.html", {"request": request, "available_models": available_models, "ratings": current_ratings, "labels": LABELS})

@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(request: Request, text: str = Form(None), file: UploadFile = File(None)):
    loaded_models = request.app.state.models; available_model_names = list(loaded_models.keys())
    input_text = ""; error_context = {"request": request, "available_models": available_model_names, "ratings": get_average_ratings(), "labels": LABELS}
    if file and file.filename:
        try: content = await file.read(); input_text = content.decode("utf-8")
        except Exception as e: return templates.TemplateResponse("index.html", {**error_context, "error": "Ошибка чтения файла"})
    elif text: input_text = text.strip()
    if not input_text: return templates.TemplateResponse("index.html", {**error_context,"error": "Введите текст или загрузите файл"})

    results = {}
    for model_name, model in loaded_models.items():
        try:
            preds = model.predict([input_text])[0]
            if not isinstance(preds, np.ndarray): preds = np.array(preds)
            if len(preds) != len(LABELS): raise ValueError("Prediction length mismatch")
            results[model_name] = {label: float(pred) for label, pred in zip(LABELS, preds)}
        except Exception as e: results[model_name] = {"error": f"Ошибка предсказания"} ; print(f"Predict Error ({model_name}): {e}")

    return templates.TemplateResponse("index.html", {"request": request, "results": results, "original_text": input_text, "ratings": get_average_ratings(), "available_models": available_model_names, "labels": LABELS})

@app.post("/rate_model_web")
async def rate_model_web(request: Request, model_name: str = Form(...), rating: int = Form(...)):
    if model_name not in request.app.state.models or not 1 <= rating <= 5: return RedirectResponse("/", status_code=303)
    add_rating(model_name, rating); return RedirectResponse("/", status_code=303)

@app.post("/predict_text")
async def predict_text(input_data: TextInput, request: Request):
    loaded_models = request.app.state.models
    if not loaded_models: raise HTTPException(status_code=503, detail="Models not loaded.")
    input_text = input_data.text.strip() if input_data.text else ""
    if not input_text: raise HTTPException(status_code=400, detail="Text input empty")
    results = {}
    for model_name, model in loaded_models.items():
        try:
            preds = model.predict([input_text])[0]
            if not isinstance(preds, np.ndarray): preds = np.array(preds)
            if len(preds) != len(LABELS): raise ValueError("Prediction length mismatch")
            results[model_name] = {label: float(pred) for label, pred in zip(LABELS, preds)}
        except Exception as e: results[model_name] = {"error": f"Prediction failed"}; print(f"Predict API Error ({model_name}): {e}")
    return results

@app.post("/predict_file")
async def predict_file(request: Request, file: UploadFile = File(...)):
    if not request.app.state.models: raise HTTPException(status_code=503, detail="Models not loaded")
    try: content = await file.read(); text = content.decode("utf-8").strip()
    except UnicodeDecodeError: raise HTTPException(status_code=400, detail="UTF-8 decode failed")
    except Exception as e: raise HTTPException(status_code=500, detail="Error processing file")
    if not text: raise HTTPException(status_code=400, detail="File content empty")
    return await predict_text(TextInput(text=text), request=request)

@app.post("/rate_model")
async def rate_model(input_data: ModelRatingInput, request: Request):
    if input_data.model_name not in request.app.state.models: raise HTTPException(status_code=404, detail=f"Model not found")
    if not 1 <= input_data.rating <= 5: raise HTTPException(status_code=400, detail="Rating must be 1-5")
    add_rating(input_data.model_name, input_data.rating); return {"message": f"Рейтинг {input_data.rating} сохранен для {input_data.model_name}"}

@app.get("/model_ratings")
async def model_ratings(): return get_average_ratings()


if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8003, reload=True)