from pydantic import BaseModel, ConfigDict

class TextInput(BaseModel):
    """Ввод пользователя через текст."""
    text: str

class ModelRatingInput(BaseModel):
    """Оценка модели."""
    model_name: str
    rating: int
    model_config = ConfigDict(protected_namespaces=())