from .base import BaseModelLoader, ModelLoadError
from .kandinsky22 import Kandinsky22Loader

# Список всех кастомных загрузчиков
CUSTOM_LOADERS = {
    "kandinsky22": Kandinsky22Loader,
}

# alias, если требуется
REGISTERED_LOADERS = CUSTOM_LOADERS


def get_loader(model_name: str):
    """
    Возвращает подходящий загрузчик по имени модели.
    Проверяет ключи в CUSTOM_LOADERS.
    """
    model_name = model_name.lower().strip()

    for key, loader_cls in CUSTOM_LOADERS.items():
        if key in model_name:
            return loader_cls()

    raise ModelLoadError(f"Не найден загрузчик для модели: {model_name}")


__all__ = [
    "BaseModelLoader",
    "ModelLoadError",
    "Kandinsky22Loader",
    "CUSTOM_LOADERS",
    "get_loader",
]
