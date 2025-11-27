import os
import json
import logging
from pathlib import Path

log = logging.getLogger("custom_loader")


class ModelLoadError(Exception):
    pass


class BaseModelLoader:
    """
    Базовый класс для локальных загрузчиков моделей.
    """

    model_name = "base"
    local_paths = []     # список локальных каталогов с моделями
    index_filename = "model_index.json"

    def __init__(self):
        self.index_path = Path(self.index_filename)

    # =======================================================
    #  Индексация моделей
    # =======================================================

    def update_model_list(self):
        """
        Сканирует локальные директории и обновляет индекс моделей.
        """
        index = {
            "local": self._scan_local()
        }

        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2)
        )

        log.info(f"[{self.model_name}] Индекс обновлён: {self.index_filename}")
        return index

    # =======================================================
    #  Загрузка модели
    # =======================================================

    def load_model(self, model_id: str):
        """
        Загружает модель ТОЛЬКО локально.
        """
        log.info(f"[{self.model_name}] Запрос загрузки: {model_id}")

        path = self.find_local_path(model_id)
        if not path:
            raise ModelLoadError(
                f"Модель '{model_id}' не найдена в локальных путях."
            )

        return self._load_from_local(path)

    # =======================================================
    #  Локальный поиск
    # =======================================================

    def _scan_local(self):
        """
        Возвращает список каталогов/файлов моделей.
        """
        results = []

        for base_path in self.local_paths:
            base = Path(base_path)
            if not base.exists():
                continue

            for entry in base.iterdir():
                if entry.is_dir():
                    results.append(entry.name)
                elif entry.suffix in (".pt", ".bin", ".safetensors"):
                    results.append(entry.name)

        return sorted(results)

    def find_local_path(self, model_id: str):
        """
        Возвращает путь к модели в local_paths.
        """
        for base_path in self.local_paths:
            base = Path(base_path)

            # 1. каталог модели
            dir_path = base / model_id
            if dir_path.exists():
                return dir_path

            # 2. файл модели
            for ext in (".pt", ".bin", ".safetensors"):
                file_path = base / f"{model_id}{ext}"
                if file_path.exists():
                    return file_path

        return None

    # =======================================================
    #  Проверка, подходит ли загрузчик для модели
    # =======================================================
    @classmethod
    def can_load(cls, weight_data, model_id: str) -> bool:
        """
        Возвращает True, если этот загрузчик может загрузить модель с указанным id.
        weight_data может быть None (для папочных моделей).
        """
        raise NotImplementedError

    # =======================================================
    #  Требуется переопределить
    # =======================================================

    def _load_from_local(self, path: Path):
        raise NotImplementedError
