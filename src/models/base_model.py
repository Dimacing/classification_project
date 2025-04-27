from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Абстрактный базовый класс модели."""

    @abstractmethod
    def build_model(self):
        """Построение архитектуры модели."""
        pass

    @abstractmethod
    def train(self, train_data, val_data):
        """Обучение модели."""
        pass

    @abstractmethod
    def predict(self, texts):
        """Предсказание по текстам."""
        pass

    @abstractmethod
    def save(self, path):
        """Сохранение модели."""
        pass

    @abstractmethod
    def load(self, path):
        """Загрузка модели."""
        pass
