from pydantic import BaseModel
from typing import List, Optional


class TextInput(BaseModel):
    """Ввод пользователя через текст."""
    text: str


class ModelRatingInput(BaseModel):
    """Оценка модели."""
    model_name: str
    rating: int  # от 1 до 5
